from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.kv_cache_no_recent import StartRecentKVCacheNoRecent
from streaming_llm.kv_cache_choice import StartRecentKVCacheChoice
from streaming_llm.kv_cache_choice_with_pos import StartRecentKVCacheChoiceWithPos
from streaming_llm.kv_cache_choice_h2o_idx import StartRecentKVCacheChoiceH2OIdx
from streaming_llm.kv_cache_choice_with_pos_mid import (
    StartRecentKVCacheChoiceWithPosMidLayers,
)

import os

def enable_streaming_llm(model, recent_use, cache_size, start_size, recent_size, compress=None):
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        # Select pos-shift implementation:
        # - default(abs): streaming_llm/pos_shift/modify_llama.py
        # - shift(rel_idx): streaming_llm/pos_shift/modify_llama_shift.py
        impl = os.environ.get("POS_SHIFT_IMPL", "abs").strip().lower()
        if impl in ("shift", "rel", "rel_idx", "relative"):
            from streaming_llm.pos_shift.modify_llama_shift import enable_llama_pos_shift_attention
        else:
            from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")
    
    print(f"Recent Use: {recent_use}")

    # compress 가 없는 경우에만 recent_use 플래그에 따라 단순 start/recent 캐시를 사용
    if compress is None:
        if recent_use == "yes":
            kv_cache = StartRecentKVCache(
                cache_size=cache_size,
                start_size=start_size,
                recent_size=recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
        else:
            kv_cache = StartRecentKVCacheNoRecent(
                cache_size=cache_size,
                start_size=start_size,
                recent_size=recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
            )
    else:
        # (attention 누적은 KV index 기준으로, compress 방식은 args.compress -> compress[name] 매핑 사용)
        kv_cache = StartRecentKVCacheChoiceWithPos(
            cache_size=cache_size,
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )
    return kv_cache
