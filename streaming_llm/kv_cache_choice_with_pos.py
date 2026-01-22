import gc
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

__all__ = ["StartRecentKVCacheChoiceWithPos"]


def slice2d(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return x[:, :, start:end, ...]


def slice3d(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return x[:, :, :, start:end, ...]


def slice1d(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCacheChoiceWithPos:
    """
    `StartRecentKVCacheChoice` 와 동일한 역할이지만, past_key_values 를 (k, v, pos)로 관리한다.

    past_key_values 형식:
        past_key_values = [
            [k_layer0, v_layer0, pos_layer0],
            [k_layer1, v_layer1, pos_layer1],
            ...
        ]

    - k, v 의 시퀀스 차원은 (k_seq_dim / v_seq_dim)으로 지정한다.
    - pos 는 "항상 마지막 차원(dim=-1)이 시퀀스"라고 가정한다.
    - 압축 시에는 sink(start_size) + recent(recent_size)는 항상 보존하고,
      중간 구간에서만 attention 기반으로 k개 토큰을 선택한다.
    """

    def __init__(
        self,
        cache_size: int = 1000,
        start_size: int = 4,
        recent_size: int = 512,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ):
        print(f"StartRecentKVCacheWithPos: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        


    @staticmethod
    def _split_layer(layer):
        # layer: (k,v) or (k,v,pos) in tuple/list form
        if len(layer) == 2:
            k, v = layer
            return k, v, None, False
        if len(layer) == 3:
            k, v, pos = layer
            return k, v, pos, True
        raise ValueError(f"unexpected past_key_values layer tuple len={len(layer)}")

    @staticmethod
    def _pack_layer(k, v, pos, has_pos: bool):
        return [k, v, pos] if has_pos else [k, v]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        keep = list(range(0, min(self.start_size, seq_len)))
        if self.recent_size > 0 and seq_len > self.recent_size:
            keep.extend(range(seq_len - self.recent_size, seq_len))
        keep = sorted(set(keep))
        keep_idx = torch.tensor(keep, dtype=torch.long, device=past_key_values[0][0].device)

        new_past = []
        keep_indices_per_layer = []
        for layer in past_key_values:
            k, v, pos, has_pos = self._split_layer(layer)
            k_new = torch.index_select(k, dim=self.k_seq_dim, index=keep_idx)
            v_new = torch.index_select(v, dim=self.v_seq_dim, index=keep_idx)
            if has_pos:
                pos_new = torch.index_select(pos, dim=-1, index=keep_idx)
            else:
                pos_new = None
            new_past.append(self._pack_layer(k_new, v_new, pos_new, has_pos))
            keep_indices_per_layer.append(keep_idx)

        self.last_keep_indices = keep_indices_per_layer
        return new_past

    def evict_for_space(self, past_key_values, num_coming: int):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        keep = list(range(0, min(self.start_size, seq_len)))
        if self.recent_size > 0 and seq_len > self.recent_size:
            start = max(seq_len - self.recent_size + num_coming, 0)
            keep.extend(range(start, seq_len))
        keep = sorted(set(keep))
        keep_idx = torch.tensor(keep, dtype=torch.long, device=past_key_values[0][0].device)

        new_past = []
        keep_indices_per_layer = []
        for layer in past_key_values:
            k, v, pos, has_pos = self._split_layer(layer)
            k_new = torch.index_select(k, dim=self.k_seq_dim, index=keep_idx)
            v_new = torch.index_select(v, dim=self.v_seq_dim, index=keep_idx)
            if has_pos:
                pos_new = torch.index_select(pos, dim=-1, index=keep_idx)
            else:
                pos_new = None
            new_past.append(self._pack_layer(k_new, v_new, pos_new, has_pos))
            keep_indices_per_layer.append(keep_idx)

        self.last_keep_indices = keep_indices_per_layer
        return new_past

    def evict_range(self, past_key_values, start: int, end: int):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len

        new_past = []
        for layer in past_key_values:
            k, v, pos, has_pos = self._split_layer(layer)
            k_new = torch.cat(
                [self.k_slice(k, 0, start), self.k_slice(k, end, seq_len)],
                dim=self.k_seq_dim,
            )
            v_new = torch.cat(
                [self.v_slice(v, 0, start), self.v_slice(v, end, seq_len)],
                dim=self.v_seq_dim,
            )
            if has_pos:
                pos_new = torch.cat([pos[..., :start], pos[..., end:seq_len]], dim=-1)
            else:
                pos_new = None
            new_past.append(self._pack_layer(k_new, v_new, pos_new, has_pos))
        return new_past

    def evict_for_space_analysis(self, past_key_values, attn_result, compress: str, k_count: int):
        """
        attention 기반으로 중간 토큰을 선택해 프루닝한다.

        - sink(start_size) / recent(recent_size)는 항상 보존
        - compress:
          - "top_k" / "moving_average" / "gaussian" / "peak_finding"
        - "layer_consistent": 레이어 평균 높고 분산 낮은 토큰 선호
        - "layer_mean": 레이어 평균(기대값)만으로 토큰 중요도를 정하고, 선택 인덱스를 전 레이어에 동일 적용
          - "layer0_only": 0번 레이어 점수만으로 선택, 선택 결과는 전 레이어 동일 적용
        """
        if past_key_values is None:
            return None
        if attn_result is None or len(attn_result) == 0:
            return past_key_values
        
        #현재 kv 캐시에 저장된 토큰 개수(시퀀스 길이)
        #지금 캐시에 토큰 L 개 있음
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # sink/recent는 항상 유지. 나머지 mid에서만 선택할 개수.
        k_mid = k_count - self.start_size - self.recent_size
        if k_mid <= 0:
            return past_key_values

        # attn_result: list of [B,H,Q,K] per layer
        attn_list = []
        for t in attn_result:
            # head mean: [B,Q,K]
            t = torch.mean(t, dim=1)
            # query mean -> [B,1,K] (Q==1이면 그대로)
            if t.shape[-2] != 1:
                t = torch.mean(t, dim=-2, keepdim=True)
                #t, _ = torch.max(t, dim=-2, keepdim=True)
            attn_list.append(t)

        min_k = min(t.shape[-1] for t in attn_list)
        if min_k <= self.start_size:
            return past_key_values

        # [L, 1, min_k]
        attn_cat = torch.cat([t[..., :min_k] for t in attn_list], dim=0)

        # mid 구간: [start_size, end_mid)
        end_mid = min_k - max(self.recent_size, 0)
        if end_mid <= self.start_size:
            return past_key_values

        attn_mid = attn_cat[:, :, self.start_size:end_mid]  # [L,1,S_mid]
        scores = attn_mid.squeeze(1)  # [L,S_mid]
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        scores_cpu = scores.detach().cpu()

        S_mid = scores_cpu.shape[-1]
        k_sel = min(k_mid, S_mid)
        if k_sel <= 0:
            return past_key_values

        if compress in ("layer_consistent", "layer_mean", "layer0_only"):
            if compress == "layer0_only":
                importance = scores_cpu[0]  # [S_mid]
            elif compress == "layer_mean":
                importance = scores_cpu.mean(dim=0)  # [S_mid]
            else:
                mean_scores = scores_cpu.mean(dim=0)
                var_scores = scores_cpu.var(dim=0, unbiased=False)
                eps = 1e-8
                norm_var = var_scores / (mean_scores.abs() + eps)
                alpha = 1.0
                importance = mean_scores / (1.0 + alpha * norm_var)

            _, top_idx = torch.topk(importance, k_sel)
            chosen_mid = top_idx.to(torch.long).tolist()
        else:
            analysis = self.analyze_attention_complete(attn_mid, k_sel, analysis_method=compress)
            if compress == "peak_finding":
                analysis = self.slice_result(analysis)
            # query_len==1 이므로 query_pos=0만 본다.
            chosen_mid = None
            # 일단 layer0 기준으로 chosen_mid를 만들고, 실제 filter는 레이어별로 수행
            # (아래 filter_past_key_values가 레이어별 chosen set 생성)

        # 레이어별로 keep index 생성 + slicing
        new_past = []
        keep_indices_per_layer: List[torch.Tensor] = []

        # recent 영역 시작 (full seq_len 기준)
        recent_start = max(seq_len - self.recent_size, 0) if self.recent_size > 0 else seq_len

        for layer_idx, layer in enumerate(past_key_values):
            k, v, pos, has_pos = self._split_layer(layer)
            # 1) mid에서 고른 인덱스(0-based in mid) -> full index
            if compress in ("layer_consistent", "layer_mean", "layer0_only"):
                mid_full = [self.start_size + x for x in chosen_mid]
            else:
                layer_map: Dict[int, List[int]] = analysis.get(layer_idx, {})
                layer_chosen = layer_map.get(0, [])
                mid_full = [self.start_size + x for x in layer_chosen]

            # sink + mid + recent 합치고 중복 제거 후 정렬
            keep_set = set(range(0, min(self.start_size, seq_len)))
            # mid는 recent와 겹치지 않게 제한
            for x in mid_full:
                if self.start_size <= x < recent_start:
                    keep_set.add(x)
            if self.recent_size > 0 and recent_start < seq_len:
                keep_set.update(range(recent_start, seq_len))

            keep_list = sorted(keep_set)
            keep_idx = torch.tensor(keep_list, dtype=torch.long, device=k.device)

            k_new = torch.index_select(k, dim=self.k_seq_dim, index=keep_idx)
            v_new = torch.index_select(v, dim=self.v_seq_dim, index=keep_idx)
            if has_pos:
                p_new = torch.index_select(pos, dim=-1, index=keep_idx)
            else:
                p_new = None

            new_past.append(self._pack_layer(k_new, v_new, p_new, has_pos))
            keep_indices_per_layer.append(keep_idx)

        self.last_keep_indices = keep_indices_per_layer

        # 메모리 정리 (원래 코드 스타일 유지)
        del attn_result
        del attn_list
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return new_past

    def slice_result(self, analysis_result):
        # peak_finding에서 layer별 길이가 다르면 최소 길이에 맞춰 잘라주는 유틸
        if not analysis_result:
            return analysis_result
        lens = []
        for layer in analysis_result.keys():
            # query_len==1 가정: query_pos=0
            lens.append(len(analysis_result[layer].get(0, [])))
        if not lens:
            return analysis_result
        min_len = min(lens)
        for layer in analysis_result.keys():
            if 0 in analysis_result[layer]:
                analysis_result[layer][0] = analysis_result[layer][0][:min_len]
        return analysis_result

    def analyze_attention_complete(
        self,
        attention_data,
        k: int,
        analysis_method: str = "gaussian",
        window_size: int = 3,
        sigma: float = 1.5,
        prominence: float = 0.001,
    ):
        """
        attention_data: torch.Tensor 또는 np.ndarray
            shape: (num_layers, query_len, key_len) 또는 (num_layers, 1, key_len)
        return:
            dict[layer_idx][query_pos] = [top_token_indices...]
        """
        if isinstance(attention_data, torch.Tensor):
            attention_data = attention_data.detach().cpu().numpy()

        num_layers, query_len, key_len = attention_data.shape
        analysis_results = {}

        if k <= 0 or key_len <= 0:
            for layer_idx in range(num_layers):
                analysis_results[layer_idx] = {q: [] for q in range(query_len)}
            return analysis_results

        for layer_idx in range(num_layers):
            layer_attention = attention_data[layer_idx]
            top_tokens_per_layer = {}

            for query_pos in range(query_len):
                scores_np = np.asarray(layer_attention[query_pos, :], dtype=np.float64)
                if scores_np.size == 0:
                    top_tokens_per_layer[query_pos] = []
                    continue

                # NOTE:
                # - inputs are attention probabilities over keys (K dim)
                # - output indices MUST be valid key indices in [0, key_len)
                # - moving_average/gaussian are smoothing methods applied on the 1D key importance curve
                if analysis_method == "moving_average":
                    # 'same' to keep 1:1 alignment with original token indices
                    if window_size <= 1 or scores_np.size < window_size:
                        smoothed = scores_np
                    else:
                        kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
                        smoothed = np.convolve(scores_np, kernel, mode="same")
                    k_eff = min(k, smoothed.size)
                    if k_eff <= 0:
                        top_tokens = []
                    else:
                        # argpartition for speed, then sort by score
                        cand = np.argpartition(-smoothed, k_eff - 1)[:k_eff]
                        cand = cand[np.argsort(-smoothed[cand])]
                        top_tokens = cand.tolist()

                elif analysis_method == "gaussian":
                    smoothed = gaussian_filter1d(scores_np, sigma=sigma)
                    k_eff = min(k, smoothed.size)
                    if k_eff <= 0:
                        top_tokens = []
                    else:
                        cand = np.argpartition(-smoothed, k_eff - 1)[:k_eff]
                        cand = cand[np.argsort(-smoothed[cand])]
                        top_tokens = cand.tolist()

                elif analysis_method == "peak_finding":
                    peaks, properties = find_peaks(scores_np, prominence=prominence)
                    prominences = properties.get("prominences", np.array([], dtype=np.float64))
                    if peaks.size == 0:
                        top_tokens = []
                    else:
                        order = np.argsort(-prominences)
                        top_peaks = peaks[order][:k]
                        top_tokens = top_peaks.tolist()

                else:  # "top_k"
                    k_eff = min(k, scores_np.size)
                    if k_eff <= 0:
                        top_tokens = []
                    else:
                        cand = np.argpartition(-scores_np, k_eff - 1)[:k_eff]
                        cand = cand[np.argsort(-scores_np[cand])]
                        top_tokens = cand.tolist()

                top_tokens_per_layer[query_pos] = top_tokens

            analysis_results[layer_idx] = top_tokens_per_layer

        return analysis_results


