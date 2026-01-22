import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(pred_tokens: List[str], gt_tokens: List[str]) -> float:
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(pred_tokens), 1)
    recall = num_same / max(len(gt_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    return f1_score(p, g)


def classification_score(prediction: str, ground_truth: str, all_classes: Any) -> float:
    # same logic as evaluation/metrics.py
    if not all_classes:
        return 0.0
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    # remove partial matches
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


DATASET_KIND = {
    # QA F1
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "triviaqa": "qa_f1",
    # classification
    "trec": "cls",
    "lsht": "cls",
    # others exist in LongBench but require rouge/jieba/etc (not supported here)
}


def score_one(dataset: str, pred: str, gts: List[str], all_classes: Any) -> float:
    kind = DATASET_KIND.get(dataset)
    if kind is None:
        raise RuntimeError(
            f"dataset '{dataset}' not supported by eval_no_deps.py. "
            "Use evaluation/eval.py with dependencies for full LongBench."
        )

    # LongBench quirks
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        pred = pred.lstrip("\n").split("\n")[0]

    best = 0.0
    for gt in gts:
        if kind == "qa_f1":
            best = max(best, qa_f1_score(pred, gt))
        elif kind == "cls":
            best = max(best, classification_score(pred, gt, all_classes=all_classes))
        else:
            raise RuntimeError(f"unknown kind: {kind}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="directory containing LongBench *.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"not a directory: {path}")

    jsonl_files = sorted([p for p in path.iterdir() if p.is_file() and p.name.endswith(".jsonl")])
    if not jsonl_files:
        raise RuntimeError(f"no *.jsonl files in: {path}")

    scores: Dict[str, float] = {}
    for fp in jsonl_files:
        dataset = fp.stem
        total = 0
        total_score = 0.0
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                total += 1
                total_score += score_one(
                    dataset,
                    d.get("pred", ""),
                    d.get("label", []),
                    d.get("all_classes", None),
                )
        scores[dataset] = round(100 * total_score / max(total, 1), 2)

    scores = dict(sorted(scores.items()))
    scores["average_score"] = round(sum(scores.values()) / len(scores), 2) if scores else 0.0

    out_path = path / "result.no_deps.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


