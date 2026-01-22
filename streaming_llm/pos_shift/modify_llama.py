import math
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    repeat_kv,
)
import types

__all__ = ["enable_llama_pos_shift_attention"]

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # 포지션 ID에 해당하는 cos, sin 값을 선택하여 적용
    cos = cos.squeeze(1).squeeze(0)  # [max_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [max_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def llama_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kv_position_ids: Optional[torch.LongTensor] = None, 
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # 1) Linear projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 2) past_k/v/pos unpack (past_key_value can be (k,v) or (k,v,pos))
    past_k = past_v = past_pos = None
    if past_key_value is not None:
        if len(past_key_value) >= 2:
            past_k, past_v = past_key_value[:2]
        if len(past_key_value) >= 3:
            past_pos = past_key_value[2]

    # 3) current absolute position ids
    # - if past_pos exists, continue from max(past_pos)+1
    # - else use root.posi_count (longbench_posi_test maintains it) or fallback to position_ids
    if past_pos is not None:
        past_pos_ids = past_pos[:, 0, :] if past_pos.dim() == 3 else past_pos
        last_pos = int(past_pos_ids.max().item())
        start_pos = last_pos + 1
        end_pos = start_pos + q_len
        cur_pos_ids = torch.arange(start_pos, end_pos, device=hidden_states.device).unsqueeze(0)
    else:
        root = getattr(self, "_root_model", None)
        if root is not None and hasattr(root, "posi_count"):
            start_pos = int(root.posi_count) - q_len
            end_pos = int(root.posi_count)
            cur_pos_ids = torch.arange(start_pos, end_pos, device=hidden_states.device).unsqueeze(0)
        elif position_ids is not None:
            cur_pos_ids = position_ids
        else:
            cur_pos_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)

    # 4) RoPE tables up to max absolute position used this step
    max_pos = int(cur_pos_ids.max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=max_pos)

    # 5) apply RoPE once using absolute pos ids
    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, cur_pos_ids)
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, cur_pos_ids)

    # 6) build absolute key_position_ids for cache (past + current)
    if past_pos is not None:
        past_pos_ids = past_pos[:, 0, :] if past_pos.dim() == 3 else past_pos
        key_position_ids = torch.cat([past_pos_ids, cur_pos_ids], dim=-1)
    else:
        key_position_ids = cur_pos_ids

    # 7) concat cached k/v (already RoPE-applied) with current
    if past_k is not None and past_v is not None:
        key_states = torch.cat([past_k, key_states], dim=2)
        value_states = torch.cat([past_v, value_states], dim=2)

    # cache stores (k_rot, v, abs_pos_ids)
    past_key_value = (key_states, value_states, key_position_ids) if use_cache else None

    kv_seq_len = key_states.shape[-2]

    # 8) Attention
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # 9) attention_mask length fix (kv_seq_len can change after pruning)
    if attention_mask is not None:
        if attention_mask.dim() == 4 and attention_mask.size(-1) != kv_seq_len:
            cur_kv = attention_mask.size(-1)
            if cur_kv > kv_seq_len:
                attention_mask = attention_mask[..., cur_kv - kv_seq_len :]
            else:
                pad_len = kv_seq_len - cur_kv
                mask_value = torch.finfo(attention_mask.dtype).min 
                attention_mask = nn.functional.pad(attention_mask, (pad_len, 0, 0, 0), value=mask_value)
        
        attn_weights = attn_weights + attention_mask

    # 10) Softmax & output
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

def enable_llama_pos_shift_attention(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            # 레이어 인덱스 주입 (디버깅용)
            import re
            layer_ids = re.findall(r'\d+', name)
            if layer_ids:
                module.layer_idx = int(layer_ids[-1])
            module._root_model = model
            
            module.forward = types.MethodType(llama_pos_shift_attention_forward, module)