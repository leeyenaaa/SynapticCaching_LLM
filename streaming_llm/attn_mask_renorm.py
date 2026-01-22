import torch


def mask_and_renorm_attn(attn_probs: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    """
    주어진 attention 분포(softmax 이후)에 대해,
    - 살려둘 key 인덱스(keep_indices) 외의 위치는 0으로 마스킹하고
    - 남은 mass 위에서 다시 정규화(renorm)한 새로운 attention 분포를 반환합니다.

    이 함수는 "원래 attention 을 최대한 보존하면서, 일부 key 위치만 프루닝했을 때
    이상적인(attention 관점에서) 분포"를 만드는 용도로 사용할 수 있습니다.

    Args:
        attn_probs (torch.Tensor):
            shape: [..., K]
            마지막 차원(K)이 key 시퀀스 길이이고, 이미 softmax 를 거친 확률 분포라고 가정합니다.
            예: [num_layers, num_heads, Q, K] 또는 [Q, K] 등.
        keep_indices (torch.Tensor):
            shape: [K_keep]
            살려둘 key 인덱스 집합입니다. (0 <= idx < K)

    Returns:
        torch.Tensor:
            attn_renorm: attn_probs 와 동일한 shape.
            - keep_indices 위치는 원래 비율을 최대한 유지하면서 다시 정규화된 값
            - 그 외 위치는 0
    """
    if keep_indices.numel() == 0:
        # 아무 것도 남기지 않는 것은 의미가 없으므로, 그대로 반환
        return attn_probs

    # attn_probs 와 같은 shape 의 mask 생성
    # 마지막 차원(K)에 대해서만 인덱싱
    # 예: shape [L, H, Q, K] 인 경우, mask 도 동일 shape
    mask = torch.zeros_like(attn_probs, dtype=torch.bool)
    # broadcasting 을 사용해 마지막 차원에만 인덱스 적용
    # view(-1, K) 형태로 펼친 뒤, 각 row 에 대해 동일한 keep_indices 를 적용
    flat_mask = mask.view(-1, mask.shape[-1])
    flat_mask[:, keep_indices] = True
    mask = flat_mask.view_as(mask)

    # 1) 선택되지 않은 위치는 0 으로 마스킹
    attn_masked = attn_probs.masked_fill(~mask, 0.0)

    # 2) 마지막 차원(K)에 대해 다시 정규화
    denom = attn_masked.sum(dim=-1, keepdim=True) + 1e-12
    attn_renorm = attn_masked / denom

    return attn_renorm


if __name__ == "__main__":
    # 간단한 데모: [Q=1, K=5] attention 에 대해 상위 2개 인덱스만 남기고 renorm
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.25, 0.15]], dtype=torch.float32)
    keep = torch.tensor([1, 3], dtype=torch.long)
    new_probs = mask_and_renorm_attn(probs, keep)
    print("orig :", probs)
    print("keep :", keep.tolist())
    print("new  :", new_probs)


