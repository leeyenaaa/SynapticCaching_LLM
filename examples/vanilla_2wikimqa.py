import argparse
import json
import os
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
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None, help="optional: limit number of samples")
    p.add_argument("--output_dir", type=str, default="results/LongBench/vanilla_2wikimqa")
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
        help="decoding implementation to use (default: greedy_loop to match longbench_posi_test.py)",
    )
    p.add_argument(
        "--truncate",
        choices=["skip", "left", "right"],
        default="skip",
        help="what to do if prompt exceeds model max length (default: skip)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_prompt(item: Dict[str, Any], template: str) -> str:
    return template.format(**item)


def encode_chat_or_plain(
    tokenizer,
    prompt_text: str,
    system_message: str,
    force_plain_prompt: bool,
) -> torch.Tensor:
    # Prefer chat template when available (e.g., Llama-3 Instruct)
    use_chat = bool(hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None))
    if force_plain_prompt:
        use_chat = False
    if use_chat:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_text},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids
        return ids
    return tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids


def get_eos_id_set(tokenizer) -> set:
    """
    Match longbench_posi_test.py behavior:
    - always include tokenizer.eos_token_id if present
    - also include <|eot_id|> if tokenizer knows it (Llama-3 Instruct often uses it)
    """
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


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "2wikimqa.jsonl"

    # LongBench template
    templates = json.load(open("streaming_llm/dataset2prompt.json", "r"))
    template = templates["2wikimqa"]

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

    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx is None:
        # conservative fallback
        max_ctx = 4096

    ds = load_dataset("zai-org/LongBench", "2wikimqa", split="test")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    # append mode so you can resume; we won't try to de-dup here
    with out_path.open("a", encoding="utf-8") as f:
        for item in tqdm(ds, desc="2wikimqa"):
            prompt_text = build_prompt(item, template)
            input_ids = encode_chat_or_plain(
                tokenizer,
                prompt_text,
                args.system_message,
                force_plain_prompt=args.force_plain_prompt,
            )

            # move to model device (device_map=auto => first parameter device is a decent proxy)
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)

            # handle too-long prompts
            if input_ids.shape[1] > max_ctx:
                if args.truncate == "skip":
                    pred_text = ""
                    record = {
                        "input": item["input"],
                        "pred": pred_text,
                        "label": item["answers"],
                        "length": item.get("length", None),
                        "dataset": item.get("dataset", "2wikimqa"),
                        "language": item.get("language", None),
                        "all_classes": item.get("all_classes", None),
                        "_id": item.get("_id", None),
                        "skipped": True,
                        "skip_reason": f"prompt_too_long: {input_ids.shape[1]} > max_ctx {max_ctx}",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
                elif args.truncate == "left":
                    input_ids = input_ids[:, -max_ctx:]
                else:  # right
                    input_ids = input_ids[:, :max_ctx]

            eos_ids = get_eos_id_set(tokenizer)

            if args.decode_impl == "generate":
                # Note: generate() takes a single eos_token_id; eot handling can differ across models.
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
                # Match longbench_posi_test.py: manual greedy decoding loop + eos/eot stop
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
                "input": item["input"],
                "pred": pred_text,
                "label": item["answers"],
                "length": item.get("length", None),
                "dataset": item.get("dataset", "2wikimqa"),
                "language": item.get("language", None),
                "all_classes": item.get("all_classes", None),
                "_id": item.get("_id", None),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


