import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def get_compressed_before_dec(record: Dict[str, Any]) -> Optional[bool]:
    # canonical
    if "compressed_before_dec" in record:
        return _parse_bool(record.get("compressed_before_dec"))

    # be robust to weird spacing in keys (e.g. "comp ressed_before_dec")
    for k, v in record.items():
        k_norm = "".join(str(k).split()).lower()  # remove all whitespace
        if k_norm == "compressed_before_dec" or k_norm == "compressedbeforedec":
            return _parse_bool(v)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="directory containing LongBench jsonl outputs")
    parser.add_argument("--out_dir", required=True, help="directory to write filtered jsonl outputs")
    parser.add_argument(
        "--keep",
        choices=["all", "true", "false"],
        default="true",
        help="which records to keep by compressed_before_dec value (default: true)",
    )
    parser.add_argument(
        "--keep_missing",
        action="store_true",
        default=True,
        help="keep records where the flag is missing/unparseable (default: True)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.name.endswith(".jsonl")])
    if not jsonl_files:
        raise RuntimeError(f"no *.jsonl files found in: {in_dir}")

    print(f"[IN ] {in_dir}")
    print(f"[OUT] {out_dir}")

    for fp in jsonl_files:
        out_fp = out_dir / fp.name
        total = kept = dropped_true = dropped_false = dropped_bad = 0

        with fp.open("r", encoding="utf-8") as fin, out_fp.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    dropped_bad += 1
                    continue

                flag = get_compressed_before_dec(rec)

                if args.keep == "true" and flag is False:
                    dropped_false += 1
                    continue
                if args.keep == "false" and flag is True:
                    dropped_true += 1
                    continue

                if flag is None and not args.keep_missing:
                    dropped_bad += 1
                    continue

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

        print(
            f"{fp.name}: total={total} kept={kept} dropped_true={dropped_true} dropped_false={dropped_false} dropped_bad={dropped_bad}"
        )

    print("\nDone.")
    print("Next: run evaluation, e.g.")
    print(f"  python evaluation/eval.py --path {str(out_dir) + os.sep}")


if __name__ == "__main__":
    main()


