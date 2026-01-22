import torch
import json
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

device='cuda'
args = parse_args()
#datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
#            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
#            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
#datasets = ['repobench-p']

datasets = [ "triviaqa", "musique", "hotpotqa", "2wikimqa", "qasper", 'samsum', 'dureader', 'lsht', 'gov_report']

datasets = ["2wikimqa"]
data = dict()
for name in datasets:
    temp = load_dataset('zai-org/LongBench', name, split='test')
    data[name] = [x for x in temp]

templates = json.load(open("streaming_llm/dataset2prompt.json", 'r'))
model, tokenizer = load(args.model_name_or_path)
max_gen_len = 1000 

past_key_values = None
if args.enable_start_recent_kv_cache: 
    kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size, recent_use=args.recent_use, 
            compress=args.compress, cache_size=args.start_size+args.recent_size)
else:
    kv_cache = None

if args.enable_pos_shift:
   from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

for name in datasets :
    with open(f"results/LongBench/streaming/{name}.jsonl", 'w') as f :
        for i, item in enumerate(tqdm(data[name])):
            # print(item.keys()) ===> 'input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id' 
            prompt = templates[name]
            inputs = prompt.format(**item)
            encodings = tokenizer.encode(inputs, add_special_tokens=True, return_tensors='pt').to(device)
           
            # truncate in middle (ref: StreamingLLM Paper Appendix D) 
            if encodings.shape[-1] > 5075 :
                half = int(7600/ 2) # max length of Llama3 = 8192 ==> 8100 ==> 8100 - 500 = 7600 
                prompt = tokenizer.decode(encodings[:, :half][0], skip_special_tokens=True)+tokenizer.decode(encodings[:, -half:][0], skip_special_tokens=True)
                encodings = tokenizer(prompt, truncation=False, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
            seq_len = encodings.shape[1]
            if kv_cache is not None:
                space_needed = seq_len + max_gen_len
                past_key_values = kv_cache.evict_for_space(past_key_values, space_needed) 
                #past_key_values = kv_cache(past_key_values) 
            
            with torch.no_grad() :
                outputs = model(input_ids=encodings, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids = [pred_token_idx.item()]
                pos = 0
                result = list() 
                for _ in range(max_gen_len - 1):
                    outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values 
                    space_needed = 1
                    if kv_cache is not None:
                        #past_key_values = kv_cache(past_key_values)
                        past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
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
                    'length': item['length'], 'dataset': item['dataset'], 'language': item['language'], 
                    'all_classes': item['all_classes'], '_id': item['_id']}
            f.write(json.dumps(for_save, ensure_ascii=False)+'\n')


f.close()
print("Done")
