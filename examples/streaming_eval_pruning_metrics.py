import os
import json
from typing import List, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from streaming_llm.utils import parse_args, load
from streaming_llm.enable_streaming_llm import enable_streaming_llm

# LongBench용 데이터셋 리스트 (요청에 따라 코드 내에 명시)
LONG_BENCH_DATASETS = [
    "triviaqa"
]

# LongBench 프롬프트 템플릿 (dataset2prompt.json 사용)
try:
    LONG_BENCH_TEMPLATES = json.load(open("streaming_llm/dataset2prompt.json", "r"))
except Exception:
    LONG_BENCH_TEMPLATES = {}

# LongBench 압축 방식 매핑 (dataset2compress.json)
try:
    LONG_BENCH_COMPRESS = json.load(open("streaming_llm/dataset2compress.json", "r"))
except Exception:
    LONG_BENCH_COMPRESS = {}

def aggregate_attention(attentions: List[torch.Tensor]) -> torch.Tensor:
    """
    Aggregate attentions across layers and heads into a single importance score per key position.
    Returns shape: [k_len] (1D vector).
    """
    # attentions: list of length num_layers
    # each tensor: [batch, num_heads, q_len, k_len]
    # We sum over layers and heads, and take the last query position
    agg = None
    for layer_attn in attentions:
        # sum over heads -> [batch, q_len, k_len]
        summed = layer_attn.sum(dim=1)
        # take last query position -> [batch, k_len]
        last_q = summed[:, -1, :]
        agg = last_q if agg is None else agg + last_q
    # batch size is 1 in typical generation
    agg = agg[0]  # [k_len]
    return agg.detach()


def update_ema_scores(
    ema_scores: torch.Tensor,
    new_scores: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Update EMA scores with new_scores; handle size growth by padding previous EMA with zeros.
    ema_scores/new_scores shape: [k_len]
    """
    if ema_scores is None:
        return new_scores.clone()
    if ema_scores.shape[0] == new_scores.shape[0]:
        return alpha * ema_scores + (1.0 - alpha) * new_scores
    # pad older EMA to new length
    old_len = ema_scores.shape[0]
    new_len = new_scores.shape[0]
    device = new_scores.device
    padded = torch.zeros(new_len, device=device, dtype=new_scores.dtype)
    padded[:old_len] = ema_scores
    padded = alpha * padded + (1.0 - alpha) * new_scores
    return padded


def compute_keep_indices(
    seq_len: int,
    ema_scores: torch.Tensor,
    start_size: int,
    recent_size: int,
    keep_total: int,
) -> torch.LongTensor:
    """
    Build keep indices containing:
      - start segment [0, start_size)
      - recent segment (last recent_size positions)
      - top remaining by EMA until keep_total
    Returns sorted indices as LongTensor on CPU.
    """
    device = ema_scores.device
    start_keep = list(range(min(start_size, seq_len)))
    recent_keep_start = max(seq_len - recent_size, 0)
    recent_keep = list(range(recent_keep_start, seq_len))
    always_keep = sorted(set(start_keep + recent_keep))

    remaining_budget = max(keep_total - len(always_keep), 0)
    if remaining_budget <= 0:
        keep = torch.tensor(always_keep[:keep_total], dtype=torch.long, device=device)
        return keep.cpu()

    candidate_start = min(start_size, seq_len)
    candidate_end = max(seq_len - recent_size, candidate_start)
    if candidate_end <= candidate_start:
        # No middle region; fill from always_keep only
        keep = torch.tensor(always_keep, dtype=torch.long, device=device)
        return keep[:keep_total].cpu()

    candidates = torch.arange(candidate_start, candidate_end, device=device)
    candidate_scores = ema_scores[candidates]
    topk = min(remaining_budget, candidates.shape[0])
    _, top_idx = torch.topk(candidate_scores, k=topk, largest=True, sorted=False)
    selected = candidates[top_idx].tolist()

    keep_all = sorted(set(always_keep + selected))
    keep = torch.tensor(keep_all[:keep_total], dtype=torch.long)
    return keep


def prune_kv_with_mapping(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    keep_indices: torch.LongTensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Prune past_key_values along sequence dimension (-2) using keep_indices.
    Returns pruned past_key_values with identical structure.
    """
    new_pkv = []
    keep_indices = keep_indices.to(past_key_values[0][0].device)
    for k, v in past_key_values:
        # k, v: [batch, num_heads, seq_len, head_dim]
        new_k = k.index_select(dim=-2, index=keep_indices)
        new_v = v.index_select(dim=-2, index=keep_indices)
        new_pkv.append((new_k, new_v))
    return tuple(new_pkv)


def recall_at_k(important_indices: torch.LongTensor, kept_indices: torch.LongTensor, k: int) -> float:
    """
    Compute Recall@K = |TopK ∩ Keep| / K
    """
    k = min(k, important_indices.numel())
    if k == 0:
        return 1.0
    topk = important_indices[:k].tolist()
    kept = set(kept_indices.tolist())
    inter = sum(1 for i in topk if i in kept)
    return inter / float(k)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()

    # Load model/tokenizer
    model, tokenizer = load(args.model_name_or_path)
    model.to(device)
    model.eval()

    max_gen_len = 1000
    max_context_len = getattr(model.config, "max_position_embeddings", 8192)

    # Reuse sizes; accept CLI overrides to match LongBench runner
    recent_use = getattr(args, "recent_use", True)
    chunk_size = getattr(args, "chunk_size", 819)
    cache_size = getattr(args, "cache_size", 6144)
    trigger_size = getattr(args, "trigger_size", 5324)
    # Match longbench_streaming_w_EMA_decChunkBiggr.py: both budgets are 50% of cache_size
    k_count = int(cache_size * 0.5)
    k_decod = int(cache_size * 0.5)

    start_size = getattr(args, "start_size", 0)
    recent_size = getattr(args, "recent_size", 0)
    ema_alpha = 0.9
    recall_k = 128  # for Recall@K (disabled until keep indices available)

    os.makedirs(args.output_dir, exist_ok=True)
    # Helper to process one sample end-to-end and write results
    def process_one_sample(inputs_text: str, label_text, f_out_handle, kv_cache, comp_way: str):
        nonlocal model, tokenizer, chunk_size, trigger_size, k_count, k_decod, device, max_gen_len
        result_tokens = []


        # Keep on CPU to reduce peak memory; move per-chunk/token to GPU lazily
        encodings = tokenizer.encode(inputs_text, add_special_tokens=False, return_tensors='pt')
        past_key_values = None
        chunks = encodings.split(chunk_size, dim=1)

        full_seq_len_local = encodings.shape[1]

        # Encoding over chunks (except last)
        for ch in range(len(chunks) - 1):
            with torch.inference_mode(), torch.cuda.amp.autocast():
                try:
                    prev_len = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
                    need_attn = (kv_cache is not None) and (prev_len + chunks[ch].shape[1] >= trigger_size)
                    c_output = model(input_ids=chunks[ch].to(device),
                                     past_key_values=past_key_values,
                                     output_attentions=need_attn,
                                     use_cache=True)
                except Exception as e:
                    print(chunks[ch].shape)
                    if past_key_values is not None:
                        print(past_key_values[0][0].shape)
                    print(f"Encoding error at chunk {ch}/{len(chunks)}: {e}")
                    return
                # Extract required refs then free outputs immediately to drop large attentions from memory
                attn_score = list(c_output.attentions) if c_output.attentions is not None else None
                past_key_values = c_output.past_key_values
                del c_output
                torch.cuda.empty_cache()
                # Prune via kv_cache when threshold reached
                if (kv_cache is not None) and (attn_score is not None) and (past_key_values[0][0].shape[-2] >= trigger_size):
                    past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, comp_way, k_count)
                    # Free attention after pruning
                    del attn_score
                    torch.cuda.empty_cache()

        # Last chunk before decoding
        encodings_last = chunks[-1]
        # Defer pruning to decoding step (where we will request attentions) to reduce peak memory

        kv_size_now_local = past_key_values[0][0].shape[-2] if past_key_values is not None else 0

        # Initial decode step
        with torch.inference_mode(), torch.cuda.amp.autocast():
            try:
                outputs = model(input_ids=encodings_last.to(device), past_key_values=past_key_values, use_cache=True)
            except Exception as e:
                print(encodings_last.shape)
                if past_key_values is not None:
                    print(past_key_values[0][0].shape)
                print(f"Decoding init error: {e}")
                return
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = [pred_token_idx.item()]
            pos = 0

        # Verification metrics accumulators
        kl_sum = 0.0
        prob_delta_sum = 0.0
        top1_same = 0
        cmp_steps = 0

        # Decoding loop
        for _ in range(max_gen_len - 1):
            # 1) Forward (pre-prune)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                cur_len = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
                need_attn = (kv_cache is not None) and (cur_len >= trigger_size)
                outputs = model(input_ids=pred_token_idx,
                                past_key_values=past_key_values,
                                output_attentions=need_attn,
                                use_cache=True)
            # Pull out only needed tensors and immediately free the module outputs to lower peak memory
            logits_before = outputs.logits[:, -1, :]
            attn_score = list(outputs.attentions) if outputs.attentions is not None else None
            past_for_prune = outputs.past_key_values
            del outputs
            torch.cuda.empty_cache()

            # 2) Prune via kv_cache using attention and comp_way
            if (kv_cache is not None) and (attn_score is not None) and (past_for_prune[0][0].shape[-2] >= trigger_size):
                pruned_past = kv_cache.evict_for_space_analysis(past_for_prune, attn_score, comp_way, k_decod)
            else:
                pruned_past = past_for_prune
            # Free attention tensors after pruning
            if attn_score is not None:
                del attn_score
            torch.cuda.empty_cache()

            # 3) Forward (post-prune) with same input to measure impact
            with torch.inference_mode(), torch.cuda.amp.autocast():
                outputs_after = model(input_ids=pred_token_idx, past_key_values=pruned_past, use_cache=True)
            logits_after = outputs_after.logits[:, -1, :]
            past_key_values = outputs_after.past_key_values
            del outputs_after
            torch.cuda.empty_cache()

            # 4) Metrics
            with torch.no_grad():
                p_before = torch.softmax(logits_before, dim=-1)
                log_q_after = torch.log_softmax(logits_after, dim=-1)
                step_kl = F.kl_div(log_q_after, p_before, reduction='batchmean').item()
                kl_sum += step_kl

                top1_before = logits_before.argmax(dim=-1).item()
                top1_after = logits_after.argmax(dim=-1).item()
                if top1_before == top1_after:
                    top1_same += 1

                prob_after = torch.softmax(logits_after, dim=-1)
                prob_delta_sum += abs(prob_after[0, top1_after].item() - p_before[0, top1_after].item())

                cmp_steps += 1

            # 5) Adopt pruned KV and continue generation from post-prune distribution
            pred_token_idx = logits_after.argmax(dim=-1).unsqueeze(1)
            generated_ids.append(pred_token_idx.item())

            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False
            ).strip().split(" ")
            now = len(generated_text) - 1
            if now > pos:
                result_tokens.append(generated_text[pos:now][-1])
                pos = now
            if pred_token_idx.item() == 128009:
                break

        # Finalize
        result_tokens.append(generated_text[pos:][-1])
        output_sentence = " ".join(result_tokens)

        if cmp_steps > 0:
            kl_avg = kl_sum / cmp_steps
            top1_stability = top1_same / cmp_steps
            prob_delta_avg = prob_delta_sum / cmp_steps
        else:
            kl_avg = 0.0
            top1_stability = 1.0
            prob_delta_avg = 0.0

        f_out_handle.write(json.dumps({
            "kv_size": f"{kv_size_now_local} / {full_seq_len_local}",
            "output": output_sentence,
            "label": label_text,
            "kl_avg": kl_avg,
            "top1_stability": top1_stability,
            "prob_delta_avg": prob_delta_avg
        }, ensure_ascii=False) + '\n')

    # Determine mode: LongBench vs single dataset
    # Support "dataset:config" input or --dataset_config
    ds_name = str(args.dataset_name)
    ds_config = getattr(args, "dataset_config", None)
    if (":" in ds_name) and (ds_config is None):
        base, cfg = ds_name.split(":", 1)
        ds_name, ds_config = base, cfg
    use_longbench = ds_name.lower() in ["longbench", "zai-org/longbench", "zai-org/longbench/"]

    if use_longbench:
        # Prepare output directory for per-dataset files
        base_dir = os.path.join(
            args.output_dir,
            f"eval_prune_metrics_longbench_{args.compress}_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_kE{k_count}_kD{k_decod}_recent{recent_use}"
        )
        print(f"Saving to directory... \n>>> {base_dir}\nConfirm(y/n) >>>", end='')
        ans = input()
        if ans != 'y':
            return
        os.makedirs(base_dir, exist_ok=True)

        # Normalize compress mapping according to args.compress (same as longbench runner)
        compress_map = dict(LONG_BENCH_COMPRESS)
        if args.compress == 'gaus':
            for key in compress_map:
                compress_map[key] = 'gaussian'
        elif args.compress == 'ema':
            for key in compress_map:
                compress_map[key] = 'moving_average'
        elif args.compress == 'peak':
            for key in compress_map:
                compress_map[key] = 'peak_finding'
        else:
            for key in compress_map:
                compress_map[key] = 'top_k'

        # Initialize kv_cache if enabled
        if getattr(args, "enable_start_recent_kv_cache", False):
            kv_cache_global = enable_streaming_llm(
                model,
                recent_use=recent_use,
                cache_size=cache_size,
                start_size=start_size,
                recent_size=recent_size,
                compress=args.compress
            )
        else:
            kv_cache_global = None

        # Load each LongBench subset and evaluate
        for name in LONG_BENCH_DATASETS:
            try:
                dataset_iter = load_dataset('zai-org/LongBench', name, split='test')
                items = [x for x in dataset_iter]
            except Exception as e:
                print(f"Failed to load LongBench subset {name}: {e}")
                continue

            out_path = os.path.join(base_dir, f"{name}.jsonl")
            with open(out_path, 'w') as f_out:
                for item in tqdm(items, desc=f"Evaluating {name}"):
                    # Build prompt via templates
                    try:
                        prompt_tmpl = LONG_BENCH_TEMPLATES.get(name, "{input}")
                        inputs_text = prompt_tmpl.format(**item)
                    except Exception:
                        # Fallback
                        inputs_text = item.get("input", "") or item.get("context", "")
                    label_text = item.get("answers", "")
                    comp_way = compress_map.get(name, 'top_k')
                    process_one_sample(inputs_text, label_text, f_out, kv_cache_global, comp_way)
        print("Done")
        return

    # Single dataset path (original behavior)
    if ".jsonl" not in ds_name:
        # Try loading with optional config
        try:
            if ds_config:
                data = load_dataset(ds_name, ds_config, split=args.split)
            else:
                data = load_dataset(ds_name, split=args.split)
        except ValueError as e:
            raise ValueError(
                f"HuggingFace 데이터셋 config가 필요합니다. 다음 중 하나로 전달하세요:\n"
                f"  1) --dataset_name {ds_name}:{'<CONFIG_NAME>'}\n"
                f"  2) --dataset_name {ds_name} --dataset_config <CONFIG_NAME>\n"
                f"오류 원문: {e}"
            )
    else:
        with open(ds_name, 'r') as f:
            data = [json.loads(d) for d in f]

    output_filepath = os.path.join(
        args.output_dir,
        f"eval_prune_metrics_{args.compress}_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_kE{k_count}_kD{k_decod}_recent{recent_use}.jsonl"
    )

    print(f"Saving to... \n>>> {output_filepath}\nConfirm(y/n) >>>", end='')
    ans = input()
    if ans != 'y':
        return

    with open(output_filepath, 'w') as f_out:
        # Default compress way outside LongBench
        if args.compress == 'gaus':
            comp_way_default = 'gaussian'
        elif args.compress == 'ema':
            comp_way_default = 'moving_average'
        elif args.compress == 'peak':
            comp_way_default = 'peak_finding'
        else:
            comp_way_default = 'top_k'
        # Initialize kv_cache if enabled
        if getattr(args, "enable_start_recent_kv_cache", False):
            kv_cache_global = enable_streaming_llm(
                model,
                recent_use=recent_use,
                cache_size=cache_size,
                start_size=start_size,
                recent_size=recent_size,
                compress=args.compress
            )
        else:
            kv_cache_global = None
        for i, item in enumerate(tqdm(data)):
            # Build prompt similar to existing example
            inp = item['prompt'].replace('\n\nQ: Can you write an appropriate summary of the above paragraphs?\nA:', '')
            inp = inp.replace('Chapter:', '')
            inputs_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful chat bot that summarizes books.<|eot_id|><|start_header_id|>user<|end_header_id|>f'Summarize the following text in about 300 words:\n{inp}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            label_text = item.get('completion', '')
            process_one_sample(inputs_text, label_text, f_out, kv_cache_global, comp_way_default)

    print("Done")


if __name__ == "__main__":
    main()


