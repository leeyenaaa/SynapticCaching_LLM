import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    # apply_rotary_pos_emb, # 원본 함수 대체됨
    repeat_kv,
)
import types

__all__ = ["enable_llama_pos_shift_attention"]


# 제공된 position ID를 기반으로 RoPE를 적용하는 헬퍼 함수
def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # x: 입력 텐서 (Query 또는 Key), shape: [bsz, num_heads, seq_len, head_dim]
    # cos, sin: 사전 계산된 RoPE 주파수, shape: [max_pos, head_dim]
    # position_ids: x에 대한 절대 위치 인덱스, shape: [bsz, seq_len]

    # position_ids가 2D [bsz, seq_len]인지 확인
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    # 필요한 경우 배치 크기 > 1 처리, 여기서는 간단하게 bsz=1로 가정
    # 음수 인덱스(예: -999 패딩)를 0으로 클램핑. 위치 0에서의 RoPE는 항등/회전 없음
    # 참고: 더 견고한 방법은 position_ids < 0 기반 마스크를 만들고
    # 마스크가 True인 곳에만 RoPE를 적용하는 것이지만, 클램핑이 더 간단함.
    # 이 클램핑이 합법적인 위치 0에 다르게 영향을 미치지 않는지 확인 필요.
    clamped_position_ids = torch.clamp(position_ids, min=0).squeeze(0) # Shape: [seq_len]
    
    # 위치에 해당하는 코사인 및 사인 값 수집
    cos_gathered = cos[clamped_position_ids].unsqueeze(0).unsqueeze(0) # Shape: [1, 1, seq_len, head_dim]
    sin_gathered = sin[clamped_position_ids].unsqueeze(0).unsqueeze(0) # Shape: [1, 1, seq_len, head_dim]

    # 필요한 경우 x 리셰이프 (예: bsz > 1이 다르게 처리된 경우)
    # x가 [bsz, num_heads, seq_len, head_dim]이고 cos/sin이 [1, 1, seq_len, head_dim]이라고 가정
    # 브로드캐스팅이 bsz 및 num_heads 차원을 처리함.
    x_embed = (x * cos_gathered) + (rotate_half(x) * sin_gathered)
    return x_embed


# 수정된 LlamaAttention 순전파 함수
def llama_pos_shift_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # 이제 past_pos + current_pos + 잠재적 패딩 포함
        # kv_position_ids: Optional[torch.LongTensor] = None, # 별도 인수로 더 이상 필요 없음
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # --- Q, K, V 투영 --- (원본 코드 동일)
    if self.config.pretraining_tp > 1:
        # 텐서 병렬 처리를 위한 KV 헤드 분할 로직 (변경 없음)
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split( (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)
        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)
        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # --- Q, K, V 리셰이프 --- (원본 코드 동일)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # --- KV 캐시 처리 ---
    # 시퀀스 길이 얻기
    kv_seq_len = key_states.shape[-2] # 현재 입력에서 온 새로운 키/값의 길이
    past_len = 0
    if past_key_value is not None:
        # 캐시에서 과거 KV의 길이 추가
        kv_seq_len += past_key_value[0].shape[-2] # 전체 KV 시퀀스 길이
        past_len = past_key_value[0].shape[-2]

    # position_ids가 2D [bsz, total_len_including_padding]인지 확인
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.shape[0] != 1:
        position_ids = position_ids.view(1, -1)

    # RoPE 테이블에 필요한 최대 위치 인덱스 결정
    # 최대 인덱스를 찾을 때 패딩 값(-999) 무시
    valid_position_ids_mask = position_ids >= 0
    if valid_position_ids_mask.any():
         max_seq_idx = torch.max(position_ids[valid_position_ids_mask]).item()
    else:
         # 패딩만 존재하거나 입력이 비어 있는 엣지 케이스 처리
         max_seq_idx = kv_seq_len -1 # 대체 처리, 조정이 필요할 수 있음

    # 필요한 최대 위치까지 RoPE 주파수 사전 계산
    cos, sin = self.rotary_emb(value_states, seq_len=max_seq_idx + 1)
    cos = cos.squeeze(1).squeeze(0) # Shape: [max_pos+1, head_dim]
    sin = sin.squeeze(1).squeeze(0) # Shape: [max_pos+1, head_dim]
    
    # Query와 Key를 위한 위치 ID 분리
    # Query 위치는 *현재* 입력 토큰에 해당 (마지막 `q_len` 위치)
    # Key 위치는 *모든* 토큰에 해당 (과거 KV + 현재 입력)
    # RoPE를 적용하기 전에 패딩을 잘라냄.

    # 패딩을 제외한 KV 캐시 + 현재 입력의 실제 길이 계산
    # 참고: 위에서 계산된 kv_seq_len은 이미 past_kv 길이를 포함함
    total_valid_len = kv_seq_len

    # position_ids를 슬라이스하여 유효한 위치만 얻음 (패딩 제외)
    # 유효하지 않은 패딩 ID(-999)가 position_ids에 포함될 수 있으므로 주의.
    # apply_rotary_pos_emb_single 내부에서 처리됨 (클램핑)
    # 슬라이싱 자체는 유효 길이까지만 수행.
    valid_position_ids_full = position_ids[:, :total_valid_len] # Shape: [bsz, total_valid_len]
    
    # 현재 쿼리 토큰에 대한 위치 추출
    # q_len은 현재 hidden_states의 길이
    query_pos = valid_position_ids_full[:, past_len:] # Shape: [bsz, q_len]
    assert query_pos.shape[1] == q_len, f"Query position length mismatch: {query_pos.shape[1]} vs {q_len}"

    # 전체 키 시퀀스(과거 + 현재)에 대한 위치
    key_pos = valid_position_ids_full # Shape: [bsz, total_valid_len]
    # Query 상태에 절대 위치를 사용하여 RoPE 적용
    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_pos)

    # --- 현재 K, V를 과거 K, V에 추가 ---
    key_states_current = key_states # 캐싱을 위해 RoPE 적용 전 현재 키 저장
    value_states_current = value_states # 캐싱을 위해 현재 값 저장
    if past_key_value is not None:
        # 과거 KV와 현재 KV 연결
        # 연결 전 키 상태 shape: [bsz, num_kv_heads, q_len, head_dim]
        # 과거 키 상태 shape: [bsz, num_kv_heads, past_len, head_dim]
        key_states = torch.cat([past_key_value[0], key_states_current], dim=2)
        value_states = torch.cat([past_key_value[1], value_states_current], dim=2)
    # else: # 과거 KV 없음, key_states와 value_states는 현재 것 그대로임
        # key_states = key_states_current
        # value_states = value_states_current

    # 캐싱을 위해 연결 후 past_key_value 튜플 업데이트
    past_key_value = (key_states, value_states) if use_cache else None

    # *전체* 키 시퀀스(과거 + 현재)에 절대 위치를 사용하여 RoPE 적용
    # 참고: 올바른 상대 위치가 절대 위치에 의해 암묵적으로 처리되도록 연결 *후* RoPE 적용.
    # key_states 텐서는 이제 shape [bsz, num_kv_heads, total_valid_len, head_dim] 가짐
    key_states_rotated = apply_rotary_pos_emb_single(key_states, cos, sin, key_pos)


    # --- 그룹화된 쿼리 어텐션 --- (원본 코드 동일)
    # n_kv_heads < n_heads인 경우 k/v 헤드 반복
    key_states_rotated = repeat_kv(key_states_rotated, self.num_key_value_groups)
    value_states_repeated = repeat_kv(value_states, self.num_key_value_groups) # 원본(회전되지 않은) 값 사용

    # --- 어텐션 계산 --- (원본 코드 동일)
    attn_weights = torch.matmul(query_states, key_states_rotated.transpose(2, 3)) / math.sqrt(self.head_dim)

    # 어텐션 가중치 크기 확인
    if attn_weights.size() != (bsz, self.num_heads, q_len, total_valid_len): # total_valid_len에 대해 확인
        raise ValueError(
            f"어텐션 가중치는 {(bsz, self.num_heads, q_len, total_valid_len)} 크기여야 하지만, "
            f"{attn_weights.size()}입니다."
        )

    # 어텐션 마스크 적용
    if attention_mask is not None:
        # 마스크 생성 방식에 따라 필요한 경우 마스크 크기 확인 조정
        expected_mask_shape = (bsz, 1, q_len, total_valid_len)
        if attention_mask.size() != expected_mask_shape:
             # 마스크가 원래 패딩 차원을 포함하는 경우 슬라이싱 시도
             try:
                 attention_mask = attention_mask[:, :, :, :total_valid_len]
                 if attention_mask.size() != expected_mask_shape:
                    raise ValueError(
                        f"어텐션 마스크는 {expected_mask_shape} 크기여야 하지만, 슬라이싱 후 {attention_mask.size()}입니다."
                    )
             except Exception as e:
                 raise ValueError(
                     f"어텐션 마스크 크기 조정 실패. 예상: {expected_mask_shape}, 실제: {attention_mask.size()}, 오류: {e}"
                 )
        attn_weights = attn_weights + attention_mask

    # Softmax 및 어텐션 출력
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_repeated) # 반복된 원본 값 사용

    # 출력 크기 확인
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output`은 {(bsz, self.num_heads, q_len, self.head_dim)} 크기여야 하지만, "
            f"{attn_output.size()}입니다."
        )

    # --- 리셰이프 및 출력 투영 --- (원본 코드 동일)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        # 텐서 병렬 처리를 위한 출력 투영 분할 (변경 없음)
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# 순전파 메서드를 몽키 패치하는 함수
def enable_llama_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()):
        # 하위 모듈에 재귀적으로 적용
        if len(list(module.children())) > 0:
            enable_llama_pos_shift_attention(module)

        # LlamaAttention 모듈 패치
        if isinstance(module, LlamaAttention):
            # 수정된 순전파 메서드를 인스턴스에 바인딩
            module.forward = types.MethodType(
                llama_pos_shift_attention_forward, module
            )
            print(f"{name}에 위치 이동 어텐션 적용됨")
