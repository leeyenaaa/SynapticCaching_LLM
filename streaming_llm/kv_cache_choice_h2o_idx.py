import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


# 시퀀스 차원(dim)에 따라 적절한 슬라이스 함수를 선택하기 위한 매핑
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCacheChoiceH2OIdx:
    """
    H2O 스타일의 누적 attention 기반 KV 캐시 압축.

    - 각 레이어별로, 캐시 인덱스(0..kv_len-1) 기준으로 attention softmax 를 step 마다 누적.
    - 압축 시점(evict_for_space_analysis)에서는
        * start_size 구간은 항상 보존 (sink)
        * recent_size 구간은 항상 보존 (recent)
        * 중간(mid) 구간은 누적 점수에 따라 상위 k 개 토큰만 남김

    past_key_values 형식:
        past_key_values = [
            [k_layer0, v_layer0, pos_layer0],
            [k_layer1, v_layer1, pos_layer1],
            ...
        ]

    여기서 pos_layerX 는 절대 position id 텐서이지만,
    이 클래스의 누적 점수는 cache index 기준으로만 관리한다.
    """

    def __init__(
        self,
        cache_size: int = 1000,
        start_size: int = 4,
        recent_size: int = 512,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        # 최근 프루닝에서 선택된 중요 토큰 인덱스(각 레이어별, sink-offset 포함)
        self.last_keep_indices: Optional[List[torch.Tensor]] = None
        # 레이어별 KV index 기준 누적 attention 점수 (tensor of shape [kv_len])
        # {layer_idx: torch.Tensor[kv_len]}
        self.accum_scores: Dict[int, torch.Tensor] = {}

    # -------- capacity 기반 truncate (attention 안 쓰는 단순 버전) --------
    def __call__(self, past_key_values):
        """
        캐시 전체 길이가 cache_size 를 넘으면
        [start 영역] + [마지막 recent_size 토큰] 만 남기고 중간을 버립니다.
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        new_past = []
        for k, v, pos in past_key_values:
            k_new = torch.cat(
                [
                    self.k_slice(k, 0, self.start_size),
                    self.k_slice(k, seq_len - self.recent_size, seq_len),
                ],
                dim=self.k_seq_dim,
            )
            v_new = torch.cat(
                [
                    self.v_slice(v, 0, self.start_size),
                    self.v_slice(v, seq_len - self.recent_size, seq_len),
                ],
                dim=self.v_seq_dim,
            )
            pos_new = torch.cat(
                [
                    pos[..., : self.start_size],
                    pos[..., seq_len - self.recent_size : seq_len],
                ],
                dim=-1,
            )
            new_past.append([k_new, v_new, pos_new])
        # 누적 스코어도 같은 인덱스로 잘라야 하지만,
        # 이 경로는 "비상용" 이라 여기서는 그대로 두고, attention 기반 경로에서만 엄밀히 관리한다.
        return new_past

    def evict_for_space(self, past_key_values, num_coming):
        """
        앞으로 들어올 토큰 수(num_coming)를 고려했을 때 cache_size 를 넘기면
        중간 구간을 제거해서 공간을 확보합니다.
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        new_past = []
        for k, v, pos in past_key_values:
            k_new = torch.cat(
                [
                    self.k_slice(k, 0, self.start_size),
                    self.k_slice(
                        k, seq_len - self.recent_size + num_coming, seq_len
                    ),
                ],
                dim=self.k_seq_dim,
            )
            v_new = torch.cat(
                [
                    self.v_slice(v, 0, self.start_size),
                    self.v_slice(
                        v, seq_len - self.recent_size + num_coming, seq_len
                    ),
                ],
                dim=self.v_seq_dim,
            )
            pos_new = torch.cat(
                [
                    pos[..., : self.start_size],
                    pos[..., seq_len - self.recent_size + num_coming : seq_len],
                ],
                dim=-1,
            )
            new_past.append([k_new, v_new, pos_new])
        return new_past

    def evict_range(self, past_key_values, start, end):
        """
        [start, end) 구간의 토큰을 통째로 제거합니다.
        """
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len

        new_past = []
        for k, v, pos in past_key_values:
            k_new = torch.cat(
                [
                    self.k_slice(k, 0, start),
                    self.k_slice(k, end, seq_len),
                ],
                dim=self.k_seq_dim,
            )
            v_new = torch.cat(
                [
                    self.v_slice(v, 0, start),
                    self.v_slice(v, end, seq_len),
                ],
                dim=self.v_seq_dim,
            )
            pos_new = torch.cat(
                [
                    pos[..., : start],
                    pos[..., end:seq_len],
                ],
                dim=-1,
            )
            new_past.append([k_new, v_new, pos_new])
        return new_past

    # -------- attention 기반 H2O-style 압축 --------
    def evict_for_space_analysis(
        self,
        past_key_values,
        attn_result,
        compress,      # 사용하지 않지만 기존 인터페이스와 호환을 위해 둠
        k_count: int,
    ):
        """
        attn_result (각 레이어의 attention softmax)를 이용해,
        레이어별 KV index 기준 누적 점수를 갱신하고,
        sink + recent 를 제외한 중간 구간을 heavy-hitter(top-k) 방식으로 압축합니다.
        """
        if past_key_values is None:
            return None
        if attn_result is None or len(attn_result) == 0:
            return past_key_values

        # 전체 seq_len 및 mid 에서 뽑을 토큰 수 계산
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        # sink + recent 보존 후 mid 에서 뽑을 개수
        mid_k = k_count - self.start_size - self.recent_size
        if mid_k <= 0:
            return past_key_values

        num_layers = len(past_key_values)

        # ---- 1) 이번 step attention 을 KV index 기준 누적 ----
        # attn_result[layer]: [B, H, q_len, kv_len_layer]
        for layer_idx in range(num_layers):
            attn_layer = attn_result[layer_idx]  # [B, H, q_len, kv_len]
            if attn_layer is None:
                continue
            # head 평균
            scores = attn_layer.mean(dim=1)  # [B, q_len, kv_len]
            # query 평균 (q_len 축)
            if scores.shape[-2] != 1:
                scores = scores.mean(dim=-2, keepdim=True)  # [B, 1, kv_len]
            # batch 차원 제거 (B=1 가정)
            scores = scores.reshape(-1, scores.shape[-1])[0]  # [kv_len]

            scores_cpu = scores.detach().cpu()
            prev = self.accum_scores.get(layer_idx)
            if prev is None or prev.shape[-1] != scores_cpu.shape[-1]:
                # 최초이거나 KV 길이가 바뀐 경우: 이번 step 값으로 초기화
                self.accum_scores[layer_idx] = scores_cpu.clone()
            else:
                self.accum_scores[layer_idx] = prev + scores_cpu

        # ---- 2) 누적 점수를 사용하여 mid 구간에 대한 pseudo-attention 텐서 구성 ----
        # mid 구간 경계 (모든 레이어에서 동일한 kv_len 을 가정)
        kv_len = past_key_values[0][0].size(self.k_seq_dim)
        if self.recent_size > 0 and kv_len > self.start_size + self.recent_size:
            mid_start = self.start_size
            mid_end = kv_len - self.recent_size
        else:
            mid_start = self.start_size
            mid_end = kv_len
        mid_len = max(0, mid_end - mid_start)
        if mid_len == 0:
            return past_key_values

        attn_accum = torch.zeros(
            (num_layers, 1, mid_len),
            dtype=torch.float32,
        )
        for layer_idx, (k, v, pos) in enumerate(past_key_values):
            layer_scores = self.accum_scores.get(layer_idx)
            if layer_scores is None:
                continue
            scores_tensor = layer_scores.to(dtype=torch.float32)
            # 길이 mismatch 시 kv_len 에 맞춤
            if scores_tensor.shape[-1] < kv_len:
                pad_len = kv_len - scores_tensor.shape[-1]
                scores_tensor = torch.cat(
                    [
                        scores_tensor,
                        torch.zeros(
                            pad_len,
                            device=scores_tensor.device,
                            dtype=scores_tensor.dtype,
                        ),
                    ],
                    dim=-1,
                )
            elif scores_tensor.shape[-1] > kv_len:
                scores_tensor = scores_tensor[..., :kv_len]

            attn_accum[layer_idx, 0, :] = scores_tensor[mid_start:mid_end]

        # 기존 분석 파이프라인 (top_k / moving_average / gaussian / peak_finding) 사용
        attn_result_np = attn_accum.detach().cpu()
        analysis_result = self.analyze_attention_complete(
            attn_result_np, mid_k, compress
        )

        if compress == "peak_finding":
            analysis_result = self.slice_result(analysis_result)

        important_kv = self.filter_past_key_values(
            past_key_values, analysis_result
        )

        new_past: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        keep_indices_per_layer: List[Optional[torch.Tensor]] = []

        for layer_idx, (k, v, pos) in enumerate(past_key_values):
            attention_sink_k = self.k_slice(k, 0, self.start_size)
            attention_sink_v = self.v_slice(v, 0, self.start_size)
            attention_sink_p = pos[..., : self.start_size]

            k_extra, v_extra, p_extra = important_kv[layer_idx]

            if self.recent_size > 0 and kv_len > self.recent_size:
                recent_k = self.k_slice(k, kv_len - self.recent_size, kv_len)
                recent_v = self.v_slice(v, kv_len - self.recent_size, kv_len)
                recent_p = pos[..., kv_len - self.recent_size : kv_len]
                k_concat = torch.cat(
                    [attention_sink_k, k_extra, recent_k], dim=self.k_seq_dim
                )
                v_concat = torch.cat(
                    [attention_sink_v, v_extra, recent_v], dim=self.v_seq_dim
                )
                p_concat = torch.cat(
                    [attention_sink_p, p_extra, recent_p], dim=-1
                )
            else:
                k_concat = torch.cat(
                    [attention_sink_k, k_extra], dim=self.k_seq_dim
                )
                v_concat = torch.cat(
                    [attention_sink_v, v_extra], dim=self.v_seq_dim
                )
                p_concat = torch.cat(
                    [attention_sink_p, p_extra], dim=-1
                )

            new_past.append([k_concat, v_concat, p_concat])
            # mid 구간의 keep 인덱스만 기록 (sink/recent 포함 X)
            keep_indices_per_layer.append(None)

        self.last_keep_indices = keep_indices_per_layer
        del past_key_values
        torch.cuda.empty_cache()
        return new_past

    def slice_result(self, analysis_result):
        # analysis_result = {layer_idx: {something}} 형태를 가정
        len_list = [len(analysis_result[layer][0]) for layer in analysis_result.keys()]
        min_len = min(len_list)

        for layer in analysis_result.keys():
            analysis_result[layer][0] = analysis_result[layer][0][:min_len]

        return analysis_result

    def analyze_attention_complete(
        self,
        attention_data,
        k,
        analysis_method="gaussian",
        window_size=3,
        sigma=1.5,
        prominence=0.001,
    ):
        """
        어텐션 맵에서 각 토큰에 가장 큰 영향을 미치는 이전 토큰 k개를 찾습니다.
        (Top-K, 이동 평균, 가우시안 필터링, 피크 찾기 방법론 선택 가능)

        Args:
            attention_data (np.ndarray 또는 torch.Tensor): 전체 어텐션 데이터.
                shape: (num_layers, query_len, key_len)
        """
        if isinstance(attention_data, torch.Tensor):
            attention_data = attention_data.detach().cpu().numpy()
        print("sh")
        num_layers, seq_len, _ = attention_data.shape
        analysis_results = {}

        for layer_idx in range(num_layers):
            layer_attention = attention_data[layer_idx]
            top_tokens_per_layer = {}

            for query_pos in range(0, seq_len):
                attention_scores = layer_attention[query_pos, :]
                top_tokens = []

                if analysis_method == "moving_average":
                    # 이동 평균(Moving Average) 방식
                    if len(attention_scores) >= window_size:
                        kernel = np.ones(window_size) / window_size
                        smoothed_scores = np.convolve(
                            attention_scores, kernel, mode="valid"
                        )
                        # 가장 점수가 높은 구간의 시작 인덱스를 찾음
                        top_indices = np.argsort(-smoothed_scores)[:k]
                        top_tokens = top_indices.tolist()
                    else:
                        # 시퀀스가 윈도우보다 짧으면 기본 top_k로 대체
                        top_indices = np.argsort(-attention_scores)[:k]
                        top_tokens = top_indices.tolist()

                elif analysis_method == "gaussian":
                    # 가우시안 필터링 방식
                    attention_scores_np = np.asarray(
                        attention_scores, dtype=np.float64
                    )
                    smoothed_scores = gaussian_filter1d(
                        attention_scores_np, sigma=sigma
                    )
                    top_indices = np.argsort(-smoothed_scores)[:k]
                    top_tokens = top_indices.tolist()

                elif analysis_method == "peak_finding":
                    # 피크 찾기 방식
                    peaks, properties = find_peaks(
                        attention_scores, prominence=0
                    )
                    peak_prominences = properties["prominences"]
                    # 중요도 순으로 정렬하여 상위 k개 선택
                    sorted_peak_indices = np.argsort(-peak_prominences)
                    top_peaks = peaks[sorted_peak_indices][:k]
                    top_tokens = top_peaks.tolist()

                else:  # 'top_k' (기본 방식)
                    top_indices = np.argsort(-attention_scores)[:k]
                    top_tokens = top_indices.tolist()

                top_tokens_per_layer[query_pos] = top_tokens

            analysis_results[layer_idx] = top_tokens_per_layer

        return analysis_results

    @torch.no_grad()
    def filter_past_key_values(self, past_key_values, analysis_results):
        """
        분석 결과(analysis_results)에 따라 각 레이어별로 중요한 토큰 인덱스를 골라
        (k, v, pos) 를 같은 인덱스로 슬라이스합니다.
        """
        if analysis_results is None:
            return past_key_values

        selected_kv = []
        keep_indices_per_layer = []
        layer_nums = analysis_results.keys()
        for layer_idx, (k, v, pos) in enumerate(past_key_values):
            if layer_idx in layer_nums:
                layer_ema = analysis_results[layer_idx]
                important_indices = set()
                for seq_idx, idx_list in layer_ema.items():
                    # Sink 만큼 인덱스 오프셋 보정 (처음에 Sink 부분 자르고 분석했으니까)
                    idx_list = [
                        x + self.start_size for x in idx_list
                    ]
                    important_indices.update(idx_list)
                important_indices = sorted(list(important_indices))
            else:
                # 해당 레이어에 분석 결과가 없다면, sink 이후를 모두 유지
                important_indices = list(
                    range(self.start_size, k.size(self.k_seq_dim))
                )

            important_indices_tensor = torch.tensor(
                important_indices, device=k.device, dtype=torch.long
            )
            k_extra = torch.index_select(
                k, dim=self.k_seq_dim, index=important_indices_tensor
            )
            v_extra = torch.index_select(
                v, dim=self.v_seq_dim, index=important_indices_tensor
            )
            p_extra = torch.index_select(
                pos, dim=-1, index=important_indices_tensor
            )

            selected_kv.append([k_extra, v_extra, p_extra])
            keep_indices_per_layer.append(important_indices_tensor)

        # 최근 프루닝에서 레이어별로 어떤 토큰 인덱스를 유지했는지 기록
        self.last_keep_indices = keep_indices_per_layer
        return selected_kv
