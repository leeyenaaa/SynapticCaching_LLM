import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set


def _get_id(record: Dict[str, Any]) -> Optional[str]:
    # canonical
    if "_id" in record:
        v = record.get("_id")
        if v is None:
            return None
        return str(v).strip()
    # be robust to whitespace in key names
    for k, v in record.items():
        k_norm = "".join(str(k).split()).lower()
        if k_norm == "_id":
            return str(v).strip()
    return None


def load_id_set(path: Path) -> Set[str]:
    """
    Accepts:
    - jsonl where each line is a dict containing "_id"
    - plain text where each line is an id string
    """
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # treat as raw id line
                ids.add(line)
                continue
            if isinstance(obj, dict):
                _id = _get_id(obj)
                if _id is not None:
                    ids.add(_id)
            elif isinstance(obj, str):
                ids.add(obj.strip())
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="directory containing LongBench jsonl outputs")
    parser.add_argument("--out_dir", required=True, help="directory to write filtered jsonl outputs")
    parser.add_argument(
        "--ids_file",
        required=True,
        help="file containing ids (plain lines) OR jsonl lines containing _id",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="optional: only filter this dataset file (e.g. 2wikimqa). If omitted, filter all *.jsonl in in_dir.",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    ids_file = Path(args.ids_file)

    if not in_dir.exists() or not in_dir.is_dir():
        raise RuntimeError(f"in_dir is not a directory: {in_dir}")
    if not ids_file.exists() or not ids_file.is_file():
        raise RuntimeError(f"ids_file is not a file: {ids_file}")

    out_dir.mkdir(parents=True, exist_ok=True)

    id_set = load_id_set(ids_file)
    if not id_set:
        raise RuntimeError(f"no ids loaded from: {ids_file}")

    if args.dataset is not None:
        jsonl_files = [in_dir / f"{args.dataset}.jsonl"]
    else:
        jsonl_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.name.endswith(".jsonl")])

    print(f"[IDS] {len(id_set)} ids loaded from {ids_file}")
    print(f"[IN ] {in_dir}")
    print(f"[OUT] {out_dir}")

    for fp in jsonl_files:
        if not fp.exists():
            print(f"[SKIP] missing: {fp.name}")
            continue

        out_fp = out_dir / fp.name
        total = kept = dropped = bad = missing_id = 0

        with fp.open("r", encoding="utf-8") as fin, out_fp.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if not isinstance(rec, dict):
                    bad += 1
                    continue
                _id = _get_id(rec)
                if _id is None:
                    missing_id += 1
                    continue
                if _id not in id_set:
                    dropped += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

        print(f"{fp.name}: total={total} kept={kept} dropped={dropped} missing_id={missing_id} bad={bad}")

    print("\nDone.")


if __name__ == "__main__":
    main()


