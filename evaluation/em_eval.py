import json
from pathlib import Path
import numpy as np

# 결과 경로 (필요하면 이름만 바꿔서 다른 실험도 같은 코드로 돌리면 됨)
base = Path("results/LongBench/streaming_ema_chunk700_cache7000_trig6180_kE3500_kD3500_recentno")
path = base / "qasper.jsonl"

def normalize_answer(s: str) -> str:
    return " ".join(s.strip().lower().split())

records = []
with path.open() as f:
    for line in f:
        d = json.loads(line)
        pred = normalize_answer(d["pred"])
        labels = [normalize_answer(x) for x in d["label"]]
        em = int(any(pred == g for g in labels))  # exact match 0/1

        records.append({
            "em": em,
            "length": d["length"],
            "kl": d["kl_avg"],
            "top1": d["top1_stability"],
            "prob_delta": d["prob_delta_avg"],
            "overlap": d["topk_overlap_avg"],
            "mass_before": d["topk_mass_before_avg"],
            "mass_after": d["topk_mass_after_avg"],
        })

print("총 샘플 수:", len(records))
print("전체 EM:", np.mean([r["em"] for r in records]))

def print_acc_by_bins(metric_name, bins):
    xs = np.array([r[metric_name] for r in records])
    ys = np.array([r["em"] for r in records])

    print(f"\n=== {metric_name} 별 EM ===")
    for lo, hi in bins:
        mask = (xs >= lo) & (xs < hi)
        if mask.sum() == 0:
            continue
        acc = ys[mask].mean()
        print(f"[{lo:.3f}, {hi:.3f})  n={mask.sum():3d}  EM={acc:.3f}")

# 예시: KL, top-k overlap, mass ratio 에 대해 확인
kl_bins = [(0,0.2),(0.2,0.5),(0.5,1.0),(1.0,2.0),(2.0,10.0)]
overlap_bins = [(0.5,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
ratio = np.array([
    r["mass_after"]/r["mass_before"] if r["mass_before"] > 0 else 0.0
    for r in records
])
for i, r in enumerate(records):
    r["mass_ratio"] = ratio[i]
ratio_bins = [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.01)]

print_acc_by_bins("kl", kl_bins)
print_acc_by_bins("overlap", overlap_bins)
print_acc_by_bins("mass_ratio", ratio_bins)

# 길이별로도 한번
len_bins = [(0,2000),(2000,4000),(4000,6000),(6000,10000)]
print_acc_by_bins("length", len_bins)
