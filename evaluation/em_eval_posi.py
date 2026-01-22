import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np

# LongBench 본 평가와 동일한 정규화 / F1을 쓰기 위해 metrics 에서 import
from metrics import normalize_answer, qa_f1_score


def strip_repeater_tail(text: str,
                        ngram_max: int = 3,
                        min_repeats: int = 5) -> str:
    """
    리피터 패턴(같은 n-gram이 여러 번 반복되는 꼬리)을 잘라낸다.
    - ngram_max: 1~ngram_max-그램까지 반복 패턴 탐지
    - min_repeats: 같은 n-gram 이 연속 min_repeats 번 이상 반복되면 그 앞에서 자름
    """
    tokens = text.split()
    L = len(tokens)
    if L == 0:
        return text

    cut_idx = L  # 기본: 안 자름

    for n in range(1, ngram_max + 1):
        if L < n * min_repeats:
            continue

        i = 0
        while i + n * min_repeats <= L:
            pattern = tokens[i:i + n]
            cnt = 1
            j = i + n
            while j + n <= L and tokens[j:j + n] == pattern:
                cnt += 1
                j += n
            if cnt >= min_repeats:
                cut_idx = min(cut_idx, i)
                break
            i += 1

    cleaned = " ".join(tokens[:cut_idx]).strip()
    return cleaned if cleaned else text.strip()


def repeater_ratio(text: str,
                   ngram_max: int = 3,
                   min_repeats: int = 5) -> float:
    """
    리피터 꼬리를 실제로 자르지는 않고,
    전체 토큰 중 '리피터 꼬리'로 판단된 구간의 비율을 반환.
    - 0.0 이면 리피터 패턴이 탐지되지 않음
    - 0.3 이면 마지막 30% 정도가 리피터 꼬리로 판단된다는 의미
    """
    tokens = text.split()
    L = len(tokens)
    if L == 0:
        return 0.0

    cut_idx = L  # 기본: 안 자름 (리피터 없음)

    for n in range(1, ngram_max + 1):
        if L < n * min_repeats:
            continue

        i = 0
        while i + n * min_repeats <= L:
            pattern = tokens[i:i + n]
            cnt = 1
            j = i + n
            while j + n <= L and tokens[j:j + n] == pattern:
                cnt += 1
                j += n
            if cnt >= min_repeats:
                cut_idx = min(cut_idx, i)
                break
            i += 1

    if cut_idx == L:
        return 0.0
    return (L - cut_idx) / L


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="평가할 jsonl 파일 경로 (예: results/LongBench/.../triviaqa.jsonl)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없음: {path}")

    data = load_jsonl(path)
    if not data:
        print("비어 있는 파일입니다.")
        return

    em_raw = []
    em_clean = []
    f1_raw = []
    f1_clean = []
    compressed_flags = []
    repeater_rates = []

    for d in data:
        raw_pred: str = d.get("pred", "")
        labels = d.get("label", [])

        # 압축 마커 제거
        has_compress = "[[COMPRESS_BEFORE_DEC]]" in raw_pred
        compressed_flags.append(int(has_compress))
        pred_wo_marker = raw_pred.replace("[[COMPRESS_BEFORE_DEC]]", " ")

        # 리피터 비율 (마커만 제거하고 측정)
        repeater_rates.append(repeater_ratio(pred_wo_marker))

        # 리피터 꼬리 제거
        pred_clean = strip_repeater_tail(pred_wo_marker)

        # LongBench metrics.normalize_answer 와 동일한 정규화 사용
        pred_raw_norm = normalize_answer(pred_wo_marker)
        pred_clean_norm = normalize_answer(pred_clean)
        gold_norm = [normalize_answer(x) for x in labels]

        # EM (정규화된 문자열 기준 완전 일치)
        em_raw.append(int(any(pred_raw_norm == g for g in gold_norm)))
        em_clean.append(int(any(pred_clean_norm == g for g in gold_norm)))

        # F1: LongBench 의 qa_f1_score 와 동일한 방식 사용 (여러 정답 중 최대 F1)
        if labels:
            f1_raw.append(max(qa_f1_score(pred_wo_marker, g) for g in labels))
            f1_clean.append(max(qa_f1_score(pred_clean, g) for g in labels))
        else:
            f1_raw.append(0.0)
            f1_clean.append(0.0)

    em_raw = np.array(em_raw, dtype=float)
    em_clean = np.array(em_clean, dtype=float)
    f1_raw = np.array(f1_raw, dtype=float)
    f1_clean = np.array(f1_clean, dtype=float)
    compressed_flags = np.array(compressed_flags, dtype=int)
    repeater_rates = np.array(repeater_rates, dtype=float)

    print("file:", str(path))
    print("N =", len(data))
    print("압축 마커 포함 샘플 수:", int(compressed_flags.sum()))
    print()
    print("EM  (raw, 마커 제거만):", float(em_raw.mean()))
    print("EM  (clean, 마커+리피터 제거):", float(em_clean.mean()))
    print("F1  (raw, 마커 제거만):", float(f1_raw.mean()))
    print("F1  (clean, 마커+리피터 제거):", float(f1_clean.mean()))
    print("Repeater ratio (mean):", float(repeater_rates.mean()))

    # 압축 여부별로 EM
    for name, mask in [
        ("non-compressed", compressed_flags == 0),
        ("compressed", compressed_flags == 1),
    ]:
        if mask.sum() == 0:
            continue
        print(f"\n[{name}] n={int(mask.sum())}")
        print("  EM raw   :", float(em_raw[mask].mean()))
        print("  EM clean :", float(em_clean[mask].mean()))
        print("  F1 raw   :", float(f1_raw[mask].mean()))
        print("  F1 clean :", float(f1_clean[mask].mean()))
        print("  Repeater ratio :", float(repeater_rates[mask].mean()))


if __name__ == "__main__":
    main()


