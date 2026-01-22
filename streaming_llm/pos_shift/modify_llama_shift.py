import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    repeat_kv,
)
import types

__all__ = ["enable_llama_pos_shift_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
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

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    # 2) past_k/v/pos unpack (past_key_value can be (k,v) or (k,v,pos))
    past_k = past_v = past_pos = None
    if past_key_value is not None:
        if len(past_key_value) >= 2:
            past_k, past_v = past_key_value[:2]
        if len(past_key_value) >= 3:
            past_pos = past_key_value[2]

    # 3) We store *UN-ROTATED* K in KV-cache.
    #    Then, for attention computation, we apply RoPE to Q/K using
    #    the KV-cache index as relative position ids:
    #    - K positions: 0 .. kv_seq_len-1
    #    - Q positions: kv_seq_len-q_len .. kv_seq_len-1
    past_len = int(past_k.shape[-2]) if past_k is not None else 0
    kv_seq_len = past_len + q_len
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # Build raw (unrotated) concatenated cache
    if past_k is not None and past_v is not None:
        key_states_cat = torch.cat([past_k, key_states], dim=2)
        value_states_cat = torch.cat([past_v, value_states], dim=2)
    else:
        key_states_cat = key_states
        value_states_cat = value_states

    # Cache stores (k_raw, v, rel_pos_ids(optional)).
    # We still keep a pos tensor for compatibility/debug, but it is "cache-local".
    key_position_ids = torch.arange(kv_seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
    past_key_value = (key_states_cat, value_states_cat, key_position_ids) if use_cache else None

    # Apply RoPE at use-time (relative to cache indices)
    key_pos = key_position_ids  # [1, kv_seq_len]
    query_pos = torch.arange(
        kv_seq_len - q_len,
        kv_seq_len,
        device=hidden_states.device,
        dtype=torch.long,
    ).unsqueeze(0)

    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, query_pos)
    key_states_rot = apply_rotary_pos_emb_single(key_states_cat, cos, sin, key_pos)


    # repeat k/v heads if n_kv_heads < n_heads
    key_states_rot = repeat_kv(key_states_rot, self.num_key_value_groups)
    value_states_cat = repeat_kv(value_states_cat, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states_rot.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        # fix attention_mask length when kv_seq_len changes (e.g., after pruning)
        if attention_mask.dim() == 4 and attention_mask.size(-1) != kv_seq_len:
            cur_kv = attention_mask.size(-1)
            if cur_kv > kv_seq_len:
                attention_mask = attention_mask[..., cur_kv - kv_seq_len :]
            else:
                pad_len = kv_seq_len - cur_kv
                mask_value = torch.finfo(attention_mask.dtype).min
                attention_mask = nn.functional.pad(attention_mask, (pad_len, 0, 0, 0), value=mask_value)
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states_cat)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_llama_pos_shift_attention(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            module._root_model = model
            module.forward = types.MethodType(llama_pos_shift_attention_forward, module)
