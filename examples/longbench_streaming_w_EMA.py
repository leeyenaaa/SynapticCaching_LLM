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
model, tokenizer = load(args.model_name_or_path)
max_gen_len = 1000 
max_context_len = model.config.max_position_embeddings

recent_use = args.recent_use
chunk_size = 819 # int(max_context_len * 0.2)
cache_size = 6144 # int(max_context_len * 0.4)
trigger_size = 5324 # int(cache_size * 0.85)

output_filepath = f"results/LongBench/streaming_gaus_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_recent{args.recent_use}"
print(f"Confirm: {chunk_size}, {cache_size}, {trigger_size} \nFile path: {output_filepath} \n(y/n)>>>", end='')
ans = input()
if ans != 'y' : exit()

os.makedirs(output_filepath, exist_ok=True)

#datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
#            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
#            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

datasets = [ "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]


data = dict()
for name in datasets:
    temp = load_dataset('zai-org/LongBench', name, split='test')
    data[name] = [x for x in temp]

attn_score = None
if args.enable_start_recent_kv_cache: 
    kv_cache = enable_streaming_llm(
            model, recent_use=recent_use, cache_size=cache_size, start_size=args.start_size, recent_size=args.recent_size)
else:
    kv_cache = None

if args.enable_pos_shift:
   from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

for name in datasets :
    with open(f"{output_filepath}/{name}.jsonl", 'w') as f :
        for i, item in enumerate(tqdm(data[name])):
            result = list() 
            # print(item.keys()) ===> 'input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id' 
            prompt = templates[name]
            inputs = prompt.format(**item)
            encodings = tokenizer.encode(inputs, add_special_tokens=True, return_tensors='pt').to(device)
            past_key_values = None

            chunks = encodings.split(chunk_size, dim=1)
            flg = 0

            for chunk in range(len(chunks)):
                if chunk == len(chunks)-1: continue
                if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:
                    past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score)

                with torch.no_grad():
                    try: 
                        c_output = model(input_ids=chunks[chunk].to(device), past_key_values=past_key_values, 
                                output_attentions=True, use_cache=True)
                    except:
                        print(chunks[chunk].shape)
                        print(past_key_values[0][0].shape)
                        result.append("Memory Out Err: {chunk[chunk].shape} / {past_key_values[0][0].shape}")
                        flg = 1 

                    past_key_values = c_output.past_key_values
                    attn_score = list(c_output.attentions)
                
            if flg != 1 :
                
                encodings = chunks[-1]
                
                if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:
                    past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score)

                seq_len = encodings.shape[1]
                 
                
                with torch.no_grad() :
                    outputs = model(input_ids=encodings, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_ids = [pred_token_idx.item()]
                    pos = 0
                    for _ in range(max_gen_len - 1):
                        outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, output_attentions=True,use_cache=True)
                        past_key_values = outputs.past_key_values 
                        attn_score = list(outputs.attentions)

                        if kv_cache is not None:
                            past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, num_coming=1)

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
print(f"Saved to {output_filepath}")
