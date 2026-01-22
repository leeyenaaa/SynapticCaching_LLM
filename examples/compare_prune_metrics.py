import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare pruning metrics between two runs (A vs B).")
    p.add_argument("--a", required=True, help="Path to JSONL file or directory (Run A).")
    p.add_argument("--b", required=True, help="Path to JSONL file or directory (Run B).")
    p.add_argument("--output_csv", default=None, help="Optional path to save comparison CSV.")
    return p.parse_args()


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def parse_kv_size(kv_size: str) -> Tuple[float, float]:
    try:
        left, right = kv_size.split("/")
        return float(left.strip()), float(right.strip())
    except Exception:
        return 0.0, 0.0


def _lenient_parse(line: str) -> Any:
    start = line.find("{")
    end = line.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(line[start : end + 1])
    return json.loads(line)


def summarize_file(jsonl_path: str) -> Dict[str, Any]:
    n = 0
    kl_sum = 0.0
    top1_sum = 0.0
    pdelta_sum = 0.0
    kept_sum = 0.0
    full_sum = 0.0
    total_lines = 0
    parsed_lines = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                try:
                    obj = _lenient_parse(line)
                except Exception:
                    continue
            parsed_lines += 1
            n += 1
            kl_sum += safe_float(obj.get("kl_avg", 0.0))
            top1_sum += safe_float(obj.get("top1_stability", 0.0))
            pdelta_sum += safe_float(obj.get("prob_delta_avg", 0.0))
            kept, full = parse_kv_size(obj.get("kv_size", "0 / 0"))
            kept_sum += kept
            full_sum += full
    if n == 0:
        return {
            "num_samples": 0,
            "total_lines": total_lines,
            "parsed_lines": parsed_lines,
            "kl_avg": None,
            "top1_stability": None,
            "prob_delta_avg": None,
            "kv_kept_avg": None,
            "full_len_avg": None,
            "keep_ratio_avg": None,
        }
    kv_kept_avg = kept_sum / n
    full_len_avg = full_sum / n
    keep_ratio_avg = (kv_kept_avg / full_len_avg) if full_len_avg > 0 else None
    return {
        "num_samples": n,
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "kl_avg": kl_sum / n,
        "top1_stability": top1_sum / n,
        "prob_delta_avg": pdelta_sum / n,
        "kv_kept_avg": kv_kept_avg,
        "full_len_avg": full_len_avg,
        "keep_ratio_avg": keep_ratio_avg,
    }


def summarize_path(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns: dataset_name -> stats dict
    If file given, key is filename without extension.
    If dir given, keys are JSONL basenames.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if os.path.isdir(path):
        files = [os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(".jsonl")]
        for fp in sorted(files):
            name = os.path.splitext(os.path.basename(fp))[0]
            out[name] = summarize_file(fp)
    else:
        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = summarize_file(path)
    return out


def write_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    cols = [
        "dataset",
        "A_kl_avg", "B_kl_avg", "Δ_kl",
        "A_top1", "B_top1", "Δ_top1",
        "A_probΔ", "B_probΔ", "Δ_probΔ",
        "A_keep_ratio", "B_keep_ratio", "Δ_keep_ratio",
        "A_num", "B_num",
    ]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join([
                r.get("dataset",""),
                str(r.get("A_kl_avg","")), str(r.get("B_kl_avg","")), str(r.get("D_kl","")),
                str(r.get("A_top1","")), str(r.get("B_top1","")), str(r.get("D_top1","")),
                str(r.get("A_prob","")), str(r.get("B_prob","")), str(r.get("D_prob","")),
                str(r.get("A_keep","")), str(r.get("B_keep","")), str(r.get("D_keep","")),
                str(r.get("A_num","")), str(r.get("B_num","")),
            ]) + "\n")


def main():
    args = parse_args()
    a = summarize_path(args.a)
    b = summarize_path(args.b)
    all_keys = sorted(set(a.keys()) | set(b.keys()))
    rows: List[Dict[str, Any]] = []
    for k in all_keys:
        A = a.get(k, {})
        B = b.get(k, {})
        A_kl = A.get("kl_avg"); B_kl = B.get("kl_avg")
        A_top = A.get("top1_stability"); B_top = B.get("top1_stability")
        A_p = A.get("prob_delta_avg"); B_p = B.get("prob_delta_avg")
        A_keep = A.get("keep_ratio_avg"); B_keep = B.get("keep_ratio_avg")
        row = {
            "dataset": k,
            "A_kl_avg": A_kl, "B_kl_avg": B_kl, "D_kl": (None if (A_kl is None or B_kl is None) else B_kl - A_kl),
            "A_top1": A_top, "B_top1": B_top, "D_top1": (None if (A_top is None or B_top is None) else B_top - A_top),
            "A_prob": A_p, "B_prob": B_p, "D_prob": (None if (A_p is None or B_p is None) else B_p - A_p),
            "A_keep": A_keep, "B_keep": B_keep, "D_keep": (None if (A_keep is None or B_keep is None) else B_keep - A_keep),
            "A_num": A.get("num_samples"), "B_num": B.get("num_samples"),
        }
        rows.append(row)

    # Pretty print to console
    for r in rows:
        print(f"{r['dataset']}: "
              f"KL A={r['A_kl_avg']} B={r['B_kl_avg']} Δ={r['D_kl']} | "
              f"Top1 A={r['A_top1']} B={r['B_top1']} Δ={r['D_top1']} | "
              f"ProbΔ A={r['A_prob']} B={r['B_prob']} Δ={r['D_prob']}")
    out_csv = args.output_csv or "compare_summary.csv"
    write_csv(rows, out_csv)
    print(f"Saved comparison to {out_csv}")


if __name__ == "__main__":
    main()


