import argparse
import json
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="분석할 jsonl 파일 경로 (예: results/LongBench/.../qasper.jsonl)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없음: {path}")

    values = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if "top1_stability" in d:
                values.append(d["top1_stability"])

    if not values:
        print("past_top1_stability 값을 가진 샘플이 없습니다.")
        return

    values = np.array(values, dtype=float)
    print("file:", str(path))
    print("n =", len(values))
    print("mean top1_stability:", float(values.mean()))
    print("min / max:", float(values.min()), "/", float(values.max()))

if __name__ == "__main__":
    main()
