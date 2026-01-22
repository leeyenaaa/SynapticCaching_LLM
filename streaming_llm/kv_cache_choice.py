import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import time
import gc
"""
어디서 OOM 발생하는지 알아보자 
"""

def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
        1: slice1d,
        2: slice2d,
        3: slice3d,
        }


class StartRecentKVCacheChoice:
    def __init__(
            self,
            cache_size=1000, 
            start_size=4,
            recent_size=512,
            k_seq_dim=2,
            v_seq_dim=2,
            ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        # 최근 프루닝에서 선택된 중요 토큰 인덱스(각 레이어별, sink-offset 포함)
        self.last_keep_indices = None

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(k, seq_len - self.recent_size, seq_len),
                            ],
                        dim=self.k_seq_dim,
                        ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            self.v_slice(v, seq_len - self.recent_size, seq_len),
                            ],
                        dim=self.v_seq_dim,
                        ),
                    ]
                for k, v in past_key_values
                ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(
                                k, seq_len - self.recent_size + num_coming, seq_len
                                ),
                            ],
                        dim=self.k_seq_dim,
                        ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            self.v_slice(
                                v, seq_len - self.recent_size + num_coming, seq_len
                                ),
                            ],
                        dim=self.v_seq_dim,
                        ),
                    ]
                for k, v in past_key_values
                ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, start),
                            self.k_slice(k, end, seq_len),
                            ],
                        dim=self.k_seq_dim,
                        ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, start),
                            self.v_slice(v, end, seq_len),
                            ],
                        dim=self.v_seq_dim,
                        ),
                    ]
                for k, v in past_key_values

                ]



    def evict_for_space_analysis(self, past_key_values, attn_result, compress, k_count):

        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        k_count = k_count - self.start_size
        # Attn Matrix 전처리 
        for layer in range(len(attn_result)) :
            attn_result[layer] = torch.mean(attn_result[layer], dim=1)  # Head 별 Average 

            if attn_result[layer].shape[-2] != 1 :
                attn_result[layer] = torch.mean(attn_result[layer], dim=-2, keepdim=True) # Input Sequence 별 Average
        
        #attn_result torch.cat(cpu_attn_list, dim=0)[:, :, self.start_size:]
        attn_result = torch.cat(attn_result, dim=0)[:, :, self.start_size:]  # Sink 제외 나머지 부분만 판별 

        # 1) layer-wise consistency 기반 압축: 모든 레이어에서 일관되게 중요한 토큰만 선택
        if compress == 'layer_consistent':
            # attn_result: [num_layers, 1, seq_len_wo_sink]
            scores = attn_result.squeeze(1)  # [L, S]
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            num_layers, seq_wo_sink = scores.shape
            if k_count <= 0 or seq_wo_sink == 0:
                return past_key_values

            # 레이어별 토큰 중요도 평균 / 분산
            scores_cpu = scores.detach().cpu()
            mean_scores = scores_cpu.mean(dim=0)              # [S]
            var_scores = scores_cpu.var(dim=0, unbiased=False)  # [S]

            eps = 1e-8
            # 정규화된 분산: mean이 작아도 variance가 크면 패널티
            norm_var = var_scores / (mean_scores.abs() + eps)
            alpha = 1.0
            importance = mean_scores / (1.0 + alpha * norm_var)

            k_sel = min(k_count, seq_wo_sink)
            # 중요도에 따라 상위 k_sel 토큰 선택 (sink 이후 기준 인덱스)
            _, top_idx = torch.topk(importance, k_sel)
            top_idx = top_idx.to(torch.long)

            new_past_key_values = []
            keep_indices_per_layer = []
            for layer_idx, (k, v) in enumerate(past_key_values):
                attention_sink_k = self.k_slice(k, 0, self.start_size)
                attention_sink_v = self.v_slice(v, 0, self.start_size)

                # sink 이후 기준 인덱스를 전체 시퀀스 기준으로 보정
                important_indices = top_idx + self.start_size  # [k_sel]
                important_indices = important_indices.to(device=k.device)

                k_extra = torch.index_select(k, dim=self.k_seq_dim, index=important_indices)
                v_extra = torch.index_select(v, dim=self.v_seq_dim, index=important_indices)

                k_concat = torch.cat([attention_sink_k, k_extra], dim=self.k_seq_dim)
                v_concat = torch.cat([attention_sink_v, v_extra], dim=self.v_seq_dim)
                new_past_key_values.append([k_concat, v_concat])
                keep_indices_per_layer.append(important_indices)

            # 최근 프루닝에서 선택된 인덱스 기록
            self.last_keep_indices = keep_indices_per_layer
            del past_key_values
            del attn_result
            torch.cuda.empty_cache()
            return new_past_key_values

        # 2) 기존 분석 방식 (top_k / moving_average / gaussian / peak_finding)
        attn_result = attn_result.detach().cpu()
        analysis_result = self.analyze_attention_complete(attn_result, k_count, compress)

        if compress == 'peak_finding':
            analysis_result = self.slice_result(analysis_result)
        important_kv = self.filter_past_key_values(past_key_values, analysis_result) 
        new_past_key_values = list()
         
        for layer_idx, (k, v) in enumerate(past_key_values) :
            attention_sink_k = self.k_slice(k, 0, self.start_size)
            attention_sink_v = self.v_slice(v, 0, self.start_size)

            k_extra = important_kv[layer_idx][0] 
            v_extra = important_kv[layer_idx][1]
            
            k_concat = [attention_sink_k, k_extra]
            v_concat = [attention_sink_v, v_extra]
            
            k_concat = torch.cat(k_concat, dim=self.k_seq_dim,)
            v_concat = torch.cat(v_concat, dim=self.v_seq_dim,)
            #k_selected = torch.cat(k_concat, dim=self.k_seq_dim,)
            #v_selected = torch.cat(v_concat, dim=self.v_seq_dim,)
            #new_past_key_values.append([k_selected, v_selected])
            new_past_key_values.append([k_concat, v_concat])
        
        del past_key_values
        del attn_result
        torch.cuda.empty_cache()
        return new_past_key_values

    def slice_result(self, analysis_result) :
        # analysis_result = {32}{1}
        len_list = [len(analysis_result[layer][0]) for layer in analysis_result.keys()] 
        min_len = min(len_list)

        for layer in analysis_result.keys() :
            analysis_result[layer][0] = analysis_result[layer][0][:min_len]

        return analysis_result  

    def analyze_attention_complete(self, attention_data, k, analysis_method='gaussian', 
            window_size=3, sigma=1.5, prominence=0.001):
        """
        어텐션 맵에서 각 토큰에 가장 큰 영향을 미치는 이전 토큰 k개를 찾습니다.
        (Top-K, 이동 평균, 가우시안 필터링, 피크 찾기 방법론 선택 가능)

        Args:
            attention_data (np.ndarray): 전체 어텐션 데이터. (num_layers, seq_len, seq_len)
            k (int): 찾고자 하는 상위 토큰의 개수.
            analysis_method (str): 분석 방법론. 'top_k', 'moving_average', 'gaussian', 'peak_finding'.
            window_size (int): 이동 평균에 사용할 윈도우 크기.
            sigma (float): 가우시안 필터링에 사용할 표준편차. 클수록 더 부드러워짐.
            prominence (float): 피크 찾기에서 피크가 주변보다 얼마나 높아야 하는지에 대한 기준.

        Returns:
            dict: 각 레이어별 분석 결과를 담은 딕셔너리.
        """
        num_layers, seq_len, _ = attention_data.shape
        analysis_results = {}

        for layer_idx in range(num_layers):
            layer_attention = attention_data[layer_idx]
            top_tokens_per_layer = {}

            for query_pos in range(0, seq_len):
                attention_scores = layer_attention[query_pos, :]
                #attention_scores = layer_attention[query_pos, :query_pos + 1]
                top_tokens = []
                raw_scores = layer_attention[query_pos, :].detach().cpu().numpy()
                total_sum = np.sum(raw_scores) + 1e-9 # Zero division 방지

                if analysis_method == 'moving_average':
                    # 이동 평균(Moving Average) 방식
                    if len(attention_scores) >= window_size:
                        kernel = np.ones(window_size) / window_size
                        smoothed_scores = np.convolve(attention_scores, kernel, mode='valid')
                        # 가장 점수가 높은 구간의 시작 인덱스를 찾음
                        top_indices = np.argsort(-smoothed_scores)[:k]
                        top_tokens = top_indices.tolist()
                    else:
                        # 시퀀스가 윈도우보다 짧으면 기본 top_k로 대체
                        top_indices = np.argsort(-attention_scores)[:k]
                        top_tokens = top_indices.tolist()

                elif analysis_method == 'gaussian':
                    # 가우시안 필터링 방식
                    attention_scores = attention_scores.detach().cpu().numpy().astype(np.float64)
                    smoothed_scores = gaussian_filter1d(attention_scores, sigma=sigma)
                    top_indices = np.argsort(-smoothed_scores)[:k]
                    top_tokens = top_indices.tolist()

                elif analysis_method == 'peak_finding':
                    # 피크 찾기 방식
                    peaks, properties = find_peaks(attention_scores, prominence=0)
                    peak_prominences = properties['prominences']
                    # 중요도 순으로 정렬하여 상위 k개 선택
                    sorted_peak_indices = np.argsort(-peak_prominences)
                    top_peaks = peaks[sorted_peak_indices][:k]
                    top_tokens = top_peaks.tolist()

                else: # 'top_k' (기본 방식)
                    top_indices = np.argsort(-attention_scores)[:k]
                    top_tokens = top_indices.tolist()
                
                top_tokens_per_layer[query_pos] = top_tokens
            analysis_results[layer_idx] = top_tokens_per_layer
        return analysis_results


    @torch.no_grad()
    def filter_past_key_values(self, past_key_values, analysis_results) :
        if analysis_results is None :
            return past_key_values

        selected_kv = list()
        keep_indices_per_layer = list()
        layer_nums = analysis_results.keys()
        for layer_idx, (k, v) in enumerate(past_key_values) :
            if layer_idx in layer_nums :
                layer_ema = analysis_results[layer_idx]
                important_indices = set()
                for seq_idx, idx_list in layer_ema.items() :
                    idx_list = [x + self.start_size for x in idx_list]  # Sink만큼 인덱스 더해주기 (처음에 Sink 부분 자르고 분석했으니까)
                    important_indices.update(idx_list)
                important_indices = sorted(list(important_indices)) 
            # 선택된 인덱스 (sink-offset 포함)를 텐서로 저장
            important_indices_tensor = torch.tensor(important_indices, device=k.device, dtype=torch.long)
            k_extra = torch.index_select(k, dim=2, index=important_indices_tensor)
            v_extra = torch.index_select(v, dim=2, index=important_indices_tensor)
            
            selected_kv.append([k_extra, v_extra])
            keep_indices_per_layer.append(important_indices_tensor)

        # 최근 프루닝에서 레이어별로 어떤 토큰 인덱스를 유지했는지 기록
        self.last_keep_indices = keep_indices_per_layer
        return selected_kv
