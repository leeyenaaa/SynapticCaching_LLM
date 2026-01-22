import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default="main")
    p.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="comma-separated LongBench dataset names, e.g. 2wikimqa,qasper,lsht,triviaqa",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="optional: limit number of samples per dataset")
    p.add_argument("--output_dir", type=str, default="results/LongBench/vanilla_longbench")
    p.add_argument("--system_message", type=str, default="You are a helpful assistant.")
    p.add_argument(
        "--force_plain_prompt",
        action="store_true",
        help="disable tokenizer chat template even if available",
    )
    p.add_argument(
        "--decode_impl",
        choices=["greedy_loop", "generate"],
        default="greedy_loop",
        help="decoding implementation (default: greedy_loop)",
    )
    p.add_argument(
        "--truncate",
        choices=["skip", "left", "right"],
        default="skip",
        help="what to do if prompt exceeds model max length (default: skip)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="append and skip already-written lines (by line count)",
    )
    return p.parse_args()


def encode_chat_or_plain(
    tokenizer,
    prompt_text: str,
    system_message: str,
    force_plain_prompt: bool,
) -> torch.Tensor:
    use_chat = bool(hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None))
    if force_plain_prompt:
        use_chat = False
    if use_chat:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids
    return tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids


def get_eos_id_set(tokenizer) -> set:
    eos_ids = set()
    try:
        if tokenizer.eos_token_id is not None:
            eos_ids.add(int(tokenizer.eos_token_id))
    except Exception:
        pass
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            eos_ids.add(int(eot))
    except Exception:
        pass
    return eos_ids


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = json.load(open("streaming_llm/dataset2prompt.json", "r"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    max_ctx = getattr(model.config, "max_position_embeddings", None) or 4096
    eos_ids = get_eos_id_set(tokenizer)

    device = next(model.parameters()).device

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not dataset_list:
        raise ValueError("--datasets is empty")

    for ds_name in dataset_list:
        if ds_name not in templates:
            raise KeyError(f"dataset '{ds_name}' not found in streaming_llm/dataset2prompt.json")

        out_path = out_dir / f"{ds_name}.jsonl"
        exist_len = count_lines(out_path) if args.resume else 0

        ds = load_dataset("zai-org/LongBench", ds_name, split="test")
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))

        if exist_len >= len(ds):
            print(f"[SKIP] {ds_name}: already has {exist_len} lines (>= {len(ds)})")
            continue

        template = templates[ds_name]
        print(f"\n=== DATASET {ds_name} (resume from {exist_len}/{len(ds)}) ===")

        with out_path.open("a", encoding="utf-8") as f:
            for item in tqdm(ds.select(range(exist_len, len(ds))), desc=ds_name):
                prompt_text = template.format(**item)
                input_ids = encode_chat_or_plain(
                    tokenizer,
                    prompt_text,
                    args.system_message,
                    force_plain_prompt=args.force_plain_prompt,
                ).to(device)

                if input_ids.shape[1] > max_ctx:
                    if args.truncate == "skip":
                        record = {
                            "input": item.get("input", ""),
                            "pred": "",
                            "label": item.get("answers", []),
                            "length": item.get("length", None),
                            "dataset": item.get("dataset", ds_name),
                            "language": item.get("language", None),
                            "all_classes": item.get("all_classes", None),
                            "_id": item.get("_id", None),
                            "skipped": True,
                            "skip_reason": f"prompt_too_long: {input_ids.shape[1]} > max_ctx {max_ctx}",
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        continue
                    if args.truncate == "left":
                        input_ids = input_ids[:, -max_ctx:]
                    else:  # right
                        input_ids = input_ids[:, :max_ctx]

                if args.decode_impl == "generate":
                    gen = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    gen_ids = gen[0, input_ids.shape[1] :]
                    pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                else:
                    out = model(input_ids=input_ids, use_cache=True)
                    past_key_values = out.past_key_values
                    pred_token_idx = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

                    generated_ids: List[int] = [int(pred_token_idx.item())]
                    for _ in range(max(args.max_new_tokens - 1, 0)):
                        out = model(
                            input_ids=pred_token_idx,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = out.past_key_values
                        pred_token_idx = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        pred_id = int(pred_token_idx.item())
                        generated_ids.append(pred_id)
                        if eos_ids and pred_id in eos_ids:
                            break
                    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                record = {
                    "input": item.get("input", ""),
                    "pred": pred_text,
                    "label": item.get("answers", []),
                    "length": item.get("length", None),
                    "dataset": item.get("dataset", ds_name),
                    "language": item.get("language", None),
                    "all_classes": item.get("all_classes", None),
                    "_id": item.get("_id", None),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()


