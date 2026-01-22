import torch

from streaming_llm.utils import load, parse_args
from streaming_llm.attn_mask_renorm import mask_and_renorm_attn


"""
간단한 실험 스크립트:

- 주어진 모델에 대해 입력 한 문장을 넣고,
- 마지막 레이어 / 마지막 query 위치에서
    - 원래 attention 으로 만든 hidden (h_base)
    - attention top-k 인덱스만 남기고 mask+renorm 한 hidden (h_ideal)
  을 비교해서,
    - cosine similarity
    - max_abs_diff
  를 출력합니다.

실행 예시:
    python examples/attn_mask_renorm_demo.py --model_name_or_path your/model
"""


def main():
    args = parse_args()
    model, tokenizer = load(args.model_name_or_path)
    model.eval()

    device = next(model.parameters()).device

    # 간단한 입력 한 문장 (원하시면 여기 바꾸셔도 됩니다)
    text = "Question: What is streaming LLM?\nAnswer:"
    enc = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(
            **enc,
            output_attentions=True,
            use_cache=True,
        )

    # HF 포맷: attentions[list] 각 원소: [batch, num_heads, Q, K]
    attn_all_layers = out.attentions
    past_key_values = out.past_key_values

    # 마지막 레이어 기준으로 실험
    layer_idx = len(attn_all_layers) - 1
    attn_layer = attn_all_layers[layer_idx][0]  # [H, Q, K], batch 0

    # head 평균 후, 마지막 query 위치 선택
    attn_mean = attn_layer.mean(dim=0)          # [Q, K]
    attn_probs_last = attn_mean[-1]             # [K]

    K = attn_probs_last.shape[-1]

    # top-k 인덱스를 "남길 인덱스"로 사용 (여기선 예시로 k=64)
    top_k = min(4, K)
    vals, idx = torch.topk(attn_probs_last, k=top_k)
    keep_indices = idx  # [top_k]

    # mask + renorm 적용
    attn_ideal = mask_and_renorm_attn(attn_probs_last, keep_indices)  # [K]

    # 같은 V 로부터 h_base, h_ideal 비교
    # past_key_values[layer_idx]: (k, v, ...) 또는 (k, v, pos, ...)
    layer_kv = past_key_values[layer_idx]
    if len(layer_kv) == 2:
        _, v = layer_kv
    else:
        _, v, _ = layer_kv

    # v: [batch, num_heads, K, D] 또는 [batch, K, num_heads, D] 일 수 있음
    # 가장 일반적인 llama 류 포맷: [B, num_heads, K, D]
    v_b0 = v[0]  # [H, K, D] 또는 [K, H, D]
    if v_b0.dim() == 3 and v_b0.shape[0] < v_b0.shape[1]:
        # [H, K, D] 로 가정
        v_mean = v_b0.mean(dim=0)  # [K, D]
    elif v_b0.dim() == 3 and v_b0.shape[0] > v_b0.shape[1]:
        # [K, H, D] 형식일 경우 transpose
        v_mean = v_b0.permute(1, 0, 2).mean(dim=1)  # [K, D]
    else:
        raise RuntimeError(f"Unexpected v shape: {v_b0.shape}")

    h_base = torch.matmul(attn_probs_last.unsqueeze(0), v_mean)   # [1, D]
    h_ideal = torch.matmul(attn_ideal.unsqueeze(0), v_mean)       # [1, D]

    sim = torch.cosine_similarity(h_base, h_ideal, dim=-1).item()
    max_diff = (h_base - h_ideal).abs().max().item()

    print("=== ATTENTION MASK+RENORM DEMO ===")
    print(f"layer_idx = {layer_idx}, K = {K}, top_k = {top_k}")
    print(f"h_base vs h_ideal cos sim = {sim:.6f}, max_abs_diff = {max_diff:.6e}")
    print(f"sum attn_base  = {float(attn_probs_last.sum()):.6f}")
    print(f"sum attn_ideal = {float(attn_ideal.sum()):.6f}")


if __name__ == "__main__":
    main()


