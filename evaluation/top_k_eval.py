import json
from pathlib import Path
import numpy as np

base = Path("results/LongBench/streaming_ema_chunk700_cache7000_trig6180_kE3500_kD3500_recentno")

for jsonl in base.glob("*.jsonl"):
    name = jsonl.stem
    kl_list = []
    top1_list = []
    prob_delta_list = []
    overlap_list = []
    mass_b_list = []
    mass_a_list = []

    with jsonl.open() as f:
        for line in f:
            d = json.loads(line)
            kl_list.append(d["kl_avg"])
            top1_list.append(d["top1_stability"])
            prob_delta_list.append(d["prob_delta_avg"])
            overlap_list.append(d["topk_overlap_avg"])
            mass_b_list.append(d["topk_mass_before_avg"])
            mass_a_list.append(d["topk_mass_after_avg"])

    print(f"[{name}] n={len(kl_list)}")
    print("  kl_avg:", np.mean(kl_list))
    print("  top1_stability:", np.mean(top1_list))
    print("  prob_delta_avg:", np.mean(prob_delta_list))
    print("  topk_overlap_avg:", np.mean(overlap_list))
    print("  topk_mass_before_avg:", np.mean(mass_b_list))
    print("  topk_mass_after_avg:", np.mean(mass_a_list))
