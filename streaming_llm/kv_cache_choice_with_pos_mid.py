import torch

from .kv_cache_choice_with_pos import StartRecentKVCacheChoiceWithPos


class StartRecentKVCacheChoiceWithPosMidLayers:
    """
    StartRecentKVCacheChoiceWithPos 를 내부적으로 사용하되,
    **attention 기반 압축(evict_for_space_analysis)** 은
    - 첫 번째 레이어
    - 마지막 레이어
    는 그대로 두고,
    - 중간 레이어들만
    에 대해서만 수행하는 래퍼 클래스.

    즉,
        [L0, L1, L2, ..., L{N-2}, L{N-1}]
    에 대해
        L0, L{N-1}  : 원본 KV 유지
        L1..L{N-2}  : StartRecentKVCacheChoiceWithPos.evict_for_space_analysis 로 압축
    를 적용한다.
    """

    def __init__(
        self,
        cache_size: int = 1000,
        start_size: int = 4,
        recent_size: int = 512,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ):
        # 내부에서 실제 압축 로직을 담당할 인스턴스
        self.inner = StartRecentKVCacheChoiceWithPos(
            cache_size=cache_size,
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )

    # capacity 기반 간단 truncate 들은 전 레이어 동일하게 적용해도 되므로
    # 그대로 inner 에 위임한다.
    def __call__(self, past_key_values):
        return self.inner(past_key_values)

    def evict_for_space(self, past_key_values, num_coming):
        return self.inner.evict_for_space(past_key_values, num_coming)

    def evict_range(self, past_key_values, start, end):
        return self.inner.evict_range(past_key_values, start, end)

    def evict_for_space_analysis(self, past_key_values, attn_result, compress, k_count):
        """
        attention 기반 압축은 **중간 레이어들만** 수행하고,
        첫 번째 / 마지막 레이어는 그대로 유지한다.
        """
        if past_key_values is None:
            return None

        num_layers = len(past_key_values)
        if num_layers == 0:
            return past_key_values

        # 레이어가 2개 이하라면, 굳이 분리할 의미가 없으므로 전체를 inner 에 맡긴다.
        if num_layers <= 2:
            return self.inner.evict_for_space_analysis(
                past_key_values, attn_result, compress, k_count
            )
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # 0번/마지막 레이어는 그대로 보존
        first_layer = past_key_values[0]
        last_layer = past_key_values[-1]
 
        # 중간 레이어들과 그에 대응하는 attention 결과만 따로 뽑아서 압축
        mid_layers = past_key_values[1:-1]
        mid_attn = attn_result[1:-1] if attn_result is not None else None

        compressed_mid = self.inner.evict_for_space_analysis(
            mid_layers, mid_attn, compress, k_count
        )

        # 다시 0번 + 압축된 중간 + 마지막 레이어 순으로 합친다.
        new_past = [first_layer]
        new_past.extend(compressed_mid)
        new_past.append(last_layer)
        return new_past


