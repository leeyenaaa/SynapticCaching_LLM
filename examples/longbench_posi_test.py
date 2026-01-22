import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.utils import parse_args, load


device='cuda'
args = parse_args()

templates = json.load(open("streaming_llm/dataset2prompt.json", 'r'))
compress = json.load(open("streaming_llm/dataset2compress.json", 'r'))

model, tokenizer = load(args.model_name_or_path)
# past_key_values 를 (k, v, pos) 형태로 받기 위해, llama 계열이면 pos_shift attention 패치를 항상 적용
# (압축 ON/OFF 비교 시에도 동일한 attention 구현을 쓰도록 맞추기 위함)
if "llama" in getattr(model.config, "model_type", ""):
    impl = os.environ.get("POS_SHIFT_IMPL", "abs").strip().lower()
    if impl in ("shift", "rel", "rel_idx", "relative"):
        from streaming_llm.pos_shift.modify_llama_shift import enable_llama_pos_shift_attention
    else:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

    enable_llama_pos_shift_attention(model)
# RoPE 절대 position 관리를 위한 전역 카운터 (인코딩 + 디코딩 전체 토큰 수)
model.posi_count = 0
max_gen_len = 1000 
max_context_len = model.config.max_position_embeddings

# Instruct/chat 모델(예: Llama-3 Instruct)은 chat template 형태로 인코딩하는 것이 정석.
# tokenizer가 chat_template/apply_chat_template를 제공하면 그 경로를 사용.
USE_CHAT_TEMPLATE = bool(
    hasattr(tokenizer, "apply_chat_template")
    and getattr(tokenizer, "chat_template", None)
)
SYSTEM_MESSAGE = "You are a helpful assistant."

# base 모델 등에서 plain prompt로 강제하고 싶으면:
#   FORCE_PLAIN_PROMPT=1 python examples/longbench_posi_test.py ...
if os.environ.get("FORCE_PLAIN_PROMPT", "0") == "1":
    USE_CHAT_TEMPLATE = False

# 디버그: model(...)이 실제로 attentions를 반환하는지 확인하고 싶을 때
# 예) DEBUG_ATTN=1 python examples/longbench_posi_test.py ...
DEBUG_ATTN = os.environ.get("DEBUG_ATTN", "0") == "1"

recent_use = args.recent_use
chunk_size = 700 #int(max_context_len * 0.15)
cache_size = 7000 #int(max_context_len * 0.440)
trigger_size = 6180 #int(max_context_len * 0.350)
recent_size = args.recent_size
k_count = int(cache_size * 0.5) 
k_decod = int(cache_size * 0.5)

output_filepath = f"results/LongBench/streaming_{args.compress}_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_kE{k_count}_kD{k_decod}_recent{args.recent_use}_noCompInDec_start_size{args.start_size}_recent_size{recent_size}_상대position_사용_test"
print(f"Confirm: {chunk_size}, {cache_size}, {trigger_size} \nFile path: {output_filepath} \n(y/n)>>>", end='')
ans = input()
if ans != 'y' : exit()

os.makedirs(output_filepath, exist_ok=True)

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

datasets = [ "triviaqa", "musique", "hotpotqa", "2wikimqa", "qasper", 'samsum', 'dureader', 'lsht', 'gov_report']

datasets = ["qasper","2wikimqa","lsht","triviaqa"]

#datasets = ["2wikimqa","qasper"]

print(datasets)
data = dict()
for name in datasets:
    temp = load_dataset('zai-org/LongBench', name, split='test')
    data[name] = [x for x in temp]

attn_score = None
if args.enable_start_recent_kv_cache: 
    kv_cache = enable_streaming_llm(
            model, recent_use=recent_use, cache_size=cache_size, start_size=args.start_size, recent_size=args.recent_size, compress=args.compress)
else:
    kv_cache = None

if args.compress == 'gaus' :
    for key in compress :
        compress[key] = 'gaussian'
elif args.compress == 'ema' :
    for key in compress :
        compress[key] = 'moving_average'
elif args.compress == 'layer_consistent':
    for key in compress:
        compress[key] = 'layer_consistent'
elif args.compress == 'layer_mean':
    for key in compress:
        compress[key] = 'layer_mean'
elif args.compress == 'layer0_only':
    for key in compress:
        compress[key] = 'layer0_only'
elif args.compress == 'peak' :
    for key in compress :
        compress[key] = 'peak_finding'
else :
    for key in compress :
        compress[key] = 'top_k'

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")


for name in datasets :
    if os.path.exists(f"{output_filepath}/{name}.jsonl"): 
        with open(f"{output_filepath}/{name}.jsonl", "r") as f:
            exist_len = [json.loads(d) for d in f]
        exist_len = len(exist_len)
        if exist_len == len(data[name]) :
            print(f"Skip {name}")
            continue
    else :
        exist_len = 0 
    
    with open(f"{output_filepath}/{name}.jsonl", 'a') as f :
        comp_way = compress[name]
        for i, item in enumerate(tqdm(data[name][exist_len:])):
            result = list() 
            kv_len = 0
            compressed_before_dec = False  # 이 샘플에서 인코딩 중 압축이 한 번이라도 일어났는지
            # sample 단위로 절대 position 카운터를 리셋해야 함 (샘플 간 누적되면 RoPE 테이블이 비정상적으로 커짐)
            model.posi_count = 0
            # print(item.keys()) ===> 'input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id' 
            prompt = templates[name]
            prompt_text = prompt.format(**item)
            if USE_CHAT_TEMPLATE:
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt_text},
                ]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encodings = tokenizer(
                    rendered,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.to(device)
            else:
                encodings = tokenizer.encode(
                    prompt_text,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)

            past_key_values = None

            chunks = encodings.split(chunk_size, dim=1)
            flg = 0
            
            for chunk in range(len(chunks)):
                if chunk == len(chunks)-1:
                    continue

                # 이번 청크 길이만큼 전역 position 카운터 증가
                model.posi_count += chunks[chunk].shape[-1]

                if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size and kv_cache is not None:
                    # attention 기반 압축: 인코딩 단계에서도 attn_result 를 사용해 프루닝
                    # evict_for_space_analysis 시그니처가 구현체마다 다를 수 있으므로
                    # 먼저 4인자 버전(past_kv, attn, compress, k_count)을 시도하고,
                    # 실패하면 3인자 버전(past_kv, attn, k_count)으로 fallback 한다.
                    try:
                        past_key_values = kv_cache.evict_for_space_analysis(
                            past_key_values, attn_score, comp_way, k_count
                        )
                    except TypeError:
                        past_key_values = kv_cache.evict_for_space_analysis(
                            past_key_values, attn_score, k_count
                        )
                    compressed_before_dec = True

                with torch.no_grad():
                    try: 
                        # attention 기반 압축을 위해 enc 단계에서도 attentions 를 얻는다.
                        c_output = model(
                            input_ids=chunks[chunk].to(device),
                            past_key_values=past_key_values, 
                            output_attentions=True,
                            use_cache=True,
                        )
                    except Exception as e:
                        print(chunks[chunk].shape)
                        if past_key_values is not None:
                            print(
                                f"Encoding error at chunk {chunk}/{len(chunks)} "
                                f"==> input {encodings.shape} / kv {past_key_values[0][0].shape}"
                            )
                            result.append(
                                f"Encoding error ({type(e).__name__}: {e}) at {chunk}/{len(chunks)} "
                                f"==> {encodings.shape} / {past_key_values[0][0].shape}"
                            )
                        else:
                            print(
                                f"Encoding error at chunk {chunk}/{len(chunks)} "
                                f"==> input {encodings.shape} / kv None"
                            )
                            result.append(
                                f"Encoding error ({type(e).__name__}: {e}) at {chunk}/{len(chunks)} "
                                f"==> {encodings.shape} / None"
                            )
                        print(f"[ENC_EXCEPTION] {type(e).__name__}: {e}")
                        flg = 1 
                        continue

                    past_key_values = c_output.past_key_values
                    attn_score = list(c_output.attentions)
            if flg != 1 :
                
                encodings = chunks[-1]
                seq_len = encodings.shape[1]
                if past_key_values is not None:
                    kv_len = past_key_values[0][0].shape

                with torch.no_grad() :
                    # 마지막 인코딩 청크 길이만큼도 전역 position 카운터 증가
                    model.posi_count += encodings.shape[-1]
                    try :
                        outputs = model(
                            input_ids=encodings,
                            past_key_values=past_key_values, 
                            output_attentions=True,
                            use_cache=True,
                        )
                    except :
                        if (
                            args.compress is not None
                            and kv_cache is not None
                            and past_key_values is not None
                            and past_key_values[0][0].shape[-2] >= trigger_size
                        ):
                            try:
                                past_key_values = kv_cache.evict_for_space_analysis(
                                    past_key_values, attn_score, comp_way, k_decod
                                )
                            except TypeError:
                                past_key_values = kv_cache.evict_for_space_analysis(
                                    past_key_values, attn_score, k_decod
                                )
                            kv_len = past_key_values[0][0].shape
                        outputs = model(
                            input_ids=encodings,
                            past_key_values=past_key_values, 
                            output_attentions=True,
                            use_cache=True,
                        )
                    past_key_values = outputs.past_key_values
                    # 첫 디코딩 토큰
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_ids = [int(pred_token_idx.item())]

                    # EOS 설정: base/instruct 모두 안전하게 종료되도록 tokenizer/model 설정을 우선 사용
                    eos_ids = set()
                    try:
                        if tokenizer.eos_token_id is not None:
                            eos_ids.add(int(tokenizer.eos_token_id))
                    except Exception:
                        pass
                    # llama3 instruct 계열에서 <|eot_id|>가 별도 종료 토큰인 경우가 있어 추가
                    try:
                        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        if isinstance(eot, int) and eot >= 0:
                            eos_ids.add(int(eot))
                    except Exception:
                        pass
                    if not eos_ids:
                        # 최후의 fallback (없으면 종료 조건이 약해짐)
                        eos_ids.add(128009)

                    for step in range(max_gen_len - 1):
                        # 생성 토큰 1개마다 전역 position 카운터 1 증가
                        model.posi_count += pred_token_idx.shape[-1]
                        outputs = model(
                            input_ids=pred_token_idx,
                            past_key_values=past_key_values,
                            output_attentions=False,
                            output_hidden_states=False,
                            use_cache=True,
                        )
                        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        pred_id = int(pred_token_idx.item())
                        generated_ids.append(pred_id)

                        # EOS 토큰을 만나면 종료 (모델/토크나이저 설정 기반)
                        if pred_id in eos_ids:
                            break

                        # 다음 step 을 위해 KV 갱신
                        past_key_values = outputs.past_key_values

                    # 마지막 step 이후의 KV
                    past_key_values = outputs.past_key_values

                # 생성 토큰 전체를 한번에 디코딩 (word-splitting incremental 방식은 매우 취약함)
                try:
                    output_sentence = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                        spaces_between_special_tokens=False,
                    ).strip()
                except Exception:
                    output_sentence = ""
            
            for_save = {
                    "input": item['input'],
                    "pred": output_sentence,
                    "label": item['answers'], 
                    "kv_size": f"{kv_len}",
                    "compressed_before_dec": compressed_before_dec,
                    "length": item['length'],
                    "dataset": item['dataset'],
                    "language": item['language'], 
                    "all_classes": item['all_classes'],
                    "_id": item['_id']}
            f.write(json.dumps(for_save, ensure_ascii=False)+'\n')


f.close()
print(f"Saved to {output_filepath}")
