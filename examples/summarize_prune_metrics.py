import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pruning metrics JSONL results.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL file or a directory containing multiple JSONL files (e.g., LongBench outputs).",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to save a CSV summary. Defaults to <input>.csv for file or <dir>/summary.csv for directory.",
    )
    return parser.parse_args()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_kv_size(kv_size: str) -> Tuple[float, float]:
    """
    kv_size format: 'kept / full'
    """
    try:
        left, right = kv_size.split("/")
        kept = float(left.strip())
        full = float(right.strip())
        return kept, full
    except Exception:
        return 0.0, 0.0


def _lenient_parse(line: str) -> Any:
    """
    Try to recover JSON from a noisy line by extracting the substring between the first '{' and last '}'.
    Returns parsed object or raises.
    """
    start = line.find("{")
    end = line.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(line[start : end + 1])
    # fallback: try strip again
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


def write_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    cols = [
        "dataset",
        "total_lines",
        "parsed_lines",
        "num_samples",
        "kl_avg",
        "top1_stability",
        "prob_delta_avg",
        "kv_kept_avg",
        "full_len_avg",
        "keep_ratio_avg",
    ]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("total_lines", "")),
                    str(row.get("parsed_lines", "")),
                    str(row.get("num_samples", "")),
                    str(row.get("kl_avg", "")),
                    str(row.get("top1_stability", "")),
                    str(row.get("prob_delta_avg", "")),
                    str(row.get("kv_kept_avg", "")),
                    str(row.get("full_len_avg", "")),
                    str(row.get("keep_ratio_avg", "")),
                ]
            ) + "\n")


def main():
    args = parse_args()
    in_path = args.input
    if not os.path.exists(in_path):
        print(f"Input path not found: {in_path}")
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    if os.path.isdir(in_path):
        jsonl_files = [os.path.join(in_path, fn) for fn in os.listdir(in_path) if fn.endswith(".jsonl")]
        jsonl_files.sort()
        if not jsonl_files:
            print(f"No JSONL files found in directory: {in_path}")
            sys.exit(1)
        for fp in jsonl_files:
            dataset = os.path.splitext(os.path.basename(fp))[0]
            stats = summarize_file(fp)
            stats["dataset"] = dataset
            rows.append(stats)
            print(f"{dataset}: {stats}")
        out_csv = args.output_csv or os.path.join(in_path, "summary.csv")
    else:
        stats = summarize_file(in_path)
        stats["dataset"] = os.path.splitext(os.path.basename(in_path))[0]
        rows.append(stats)
        print(f"{stats['dataset']}: {stats}")
        base, _ = os.path.splitext(in_path)
        out_csv = args.output_csv or f"{base}.csv"

    write_csv(rows, out_csv)
    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    main()


