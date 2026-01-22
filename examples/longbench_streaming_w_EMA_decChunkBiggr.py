import torch
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
max_gen_len = 1000 
max_context_len = model.config.max_position_embeddings

recent_use = args.recent_use
chunk_size = 819 #int(max_context_len * 0.15)
cache_size = 6144 #int(max_context_len * 0.440)
trigger_size = 5324 #int(max_context_len * 0.350)
k_count = int(cache_size * 0.5) 
k_decod = int(cache_size * 0.8)

output_filepath = f"results/LongBench/streaming_{args.compress}_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_kE{k_count}_kD{k_decod}_recent{args.recent_use}_noCompInDec"
print(f"Confirm: {chunk_size}, {cache_size}, {trigger_size} \nFile path: {output_filepath} \n(y/n)>>>", end='')
ans = input()
if ans != 'y' : exit()

os.makedirs(output_filepath, exist_ok=True)

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

datasets = [ "triviaqa", "musique", "hotpotqa", "2wikimqa", "qasper", 'samsum', 'dureader', 'lsht', 'gov_report']

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

if args.enable_pos_shift:
   from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

if args.compress == 'gaus' :
    for key in compress :
        compress[key] = 'gaussian'
elif args.compress == 'ema' :
    for key in compress :
        compress[key] = 'moving_average'
elif args.compress == 'peak' :
    for key in compress :
        compress[key] = 'peak_finding'
else :
    for key in compress :
        compress[key] = 'top_k'

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

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
            # print(item.keys()) ===> 'input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id' 
            prompt = templates[name]
            inputs = prompt.format(**item)
            encodings = tokenizer.encode(inputs, add_special_tokens=True, return_tensors='pt').to(device)
            past_key_values = None

            chunks = encodings.split(chunk_size, dim=1)
            flg = 0
            
            attn_flg = False
            for chunk in range(len(chunks)):
                if chunk == len(chunks)-1: continue
                if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:
                    past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, comp_way, k_count)
                    attn_flg=True

                with torch.no_grad():
                    try: 
                        c_output = model(input_ids=chunks[chunk].to(device), past_key_values=past_key_values, 
                                output_attentions=True, use_cache=True)
                    except:
                        print(chunks[chunk].shape)
                        print(f"Memory Out Err in encoding process: {chunk}/{len(chunks)} ==> {encodings.shape} / {past_key_values[0][0].shape}")
                        result.append(f"Memory Out Err in encoding process: {chunk}/{len(chunks)} ==> {encodings.shape} / {past_key_values[0][0].shape}")
                        flg = 1 

                    past_key_values = c_output.past_key_values
                    attn_score = list(c_output.attentions)  
            if flg != 1 :
                
                encodings = chunks[-1]
                seq_len = encodings.shape[1]
                if past_key_values == None :
                    kv_len = 0
                else : kv_len = past_key_values[0][0].shape

                with torch.no_grad() :
                    try :
                        outputs = model(input_ids=encodings, past_key_values=past_key_values, 
                                output_attentions=True, use_cache=True)
                    except :
                        past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, comp_way, k_decod)
                        kv_len = past_key_values[0][0].shape
                        outputs = model(input_ids=encodings, past_key_values=past_key_values, 
                                output_attentions=True, use_cache=True)
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_ids = [pred_token_idx.item()]
                    pos = 0
                    for _ in range(max_gen_len - 1):
                        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, output_attentions=False,use_cache=True)
                        past_key_values = outputs.past_key_values 

#                        if kv_cache is not None and past_key_values[0][0].shape[-2] >= trigger_size:
#                            past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, comp_way, k_decod)

                        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        generated_ids.append(pred_token_idx.item())
                        generated_text = (
                                tokenizer.decode(generated_ids, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True, spaces_between_special_tokens=False).strip().split(" "))
                        now = len(generated_text) - 1
                        if now > pos :
                            result.append(generated_text[pos:now][-1])
                            pos = now
                        if pred_token_idx == 128009:
                            break
            

                try: 
                    result.append(generated_text[pos:][-1])
                except :
                    result.append("")
            
            output_sentence = " ".join(result)
            
            for_save = {
                    "input": item['input'], "pred": output_sentence, 'label': item['answers'], 
                    "kv_size": f"{kv_len}", 
                    'length': item['length'], 'dataset': item['dataset'], 'language': item['language'], 
                    'all_classes': item['all_classes'], '_id': item['_id']}
            f.write(json.dumps(for_save, ensure_ascii=False)+'\n')


f.close()
print(f"Saved to {output_filepath}")
