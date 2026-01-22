import time
import torch
import json
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.kv_cache_no_recent import StartRecentKVCacheNoRecent
from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import parse_args, load

device='cuda'
args = parse_args()
if ".jsonl" not in args.dataset_name:
    data = load_dataset(args.dataset_name, split=args.split)
else :
    with open(args.dataset_name, 'r') as f:
        data = [json.loads(d) for d in f]
    
model, tokenizer = load(args.model_name_or_path)
max_gen_len = 1000 
max_context_len = model.config.max_position_embeddings

recent_use = args.recent_use
chunk_size = 1064 #int(max_context_len * 0.13)
cache_size = 4096 #int(max_context_len * 0.7)
trigger_size = 3276 #int(max_context_len * 0.6)
k_count =  1064 #chunk_size 
k_decod = 2867 #int(cache_size * 0.7)
comp_way = args.compress

#print(f"Confirm: {chunk_size}, {cache_size}, {trigger_size} >>>", end='')
attn_score = None
if args.enable_start_recent_kv_cache: 
    kv_cache = enable_streaming_llm(
            model, recent_use=recent_use, compress=comp_way, cache_size=cache_size, start_size=args.start_size, recent_size=args.recent_size)
else:
    kv_cache = None

if args.enable_pos_shift:
   from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

output_filepath = f"results/booksum/encode/streaming_{comp_way}_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_kE{k_count}_kD{k_decod}_recent{args.recent_use}.jsonl"
os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

print(f"Saving to... \n>>> {output_filepath}\nConfirm(y/n) >>>", end='')
ans = input()
if ans != 'y':
    exit()

with open(f"{output_filepath}", 'w') as f :
    for i, item in enumerate(tqdm(data[:100])):
        result = list() 
        inputs = item['prompt'].replace('\n\nQ: Can you write an appropriate summary of the above paragraphs?\nA:', '')
        inputs = item['prompt'].replace('Chapter:', '')
        inputs = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful chat bot that summarizes books.<|eot_id|><|start_header_id|>user<|end_header_id|>f'Summarize the following text in about 300 words:\n{inputs}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        encodings = tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt').to(device)
        past_key_values = None 
        # If Overflow: Chunking + Encoding w/RecentKV, EMA 
        chunks = encodings.split(chunk_size, dim=1)
        compress_count = 0
        
        chunking_time = list()
        compress_time = list()

        start_time = time.time()
        for chunk in range(len(chunks)) :
            if chunk == len(chunks)-1: continue
            if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:
                # (7600/8) * 6650? 
                # Pruning with about EMA score
                comp_start = time.time()
                past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, comp_way, k_count)
                compress_count += 1
                comp_time = time.time() - comp_start

            else :
                comp_time = 0
            compress_time.append(comp_time)
            
            with torch.no_grad():
                token_start = time.time()
                try :
                    c_output = model(input_ids=chunks[chunk].to(device), past_key_values=past_key_values, output_attentions=True, use_cache=True)
                token_time = time.time() - token_start
                except:
                    print(chunks[chunk].shape)
                    print(past_key_values[0][0].shape)
                    result.append(f"Memory Out Err in encoding process: {chunk}/{len(chunks)} ==> {encodings.shape} / {past_key_values[0][0].shape}")
                    print(f"Memory Out Err in encoding process: {chunk}/{len(chunks)} ==> {encodings.shape} / {past_key_values[0][0].shape}")
                    token_time = 0
                    break

                chunking_time.append(token_time)
                past_key_values = c_output.past_key_values
                attn_score = list(c_output.attentions)

        full_seq_len = encodings.shape[1]
        encodings = chunks[-1]
        final_compress = 'no'

        if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:
            # (7600/8) * 6650? 
            # Pruning with about EMA score
            comp_start = time.time()
            past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, comp_way, k_decod)
            comp_time = time.time() - comp_start
            compress_time.append(comp_time)

            compress_count += 1
            final_compress = 'yes'
            
        
        end_time = time.time()
        total_time = end_time - start_time 

        avg_chunking = sum(chunking_time) / len(chunking_time)
        avg_compress = sum(compress_time) / len(compress_time)

        seq_len = encodings.shape[1]
        if past_key_values is not None : kv_size = past_key_values[0][0].shape[-2]
        else : kv_size = 0
        

        f.write(json.dumps({
                                #"input": inputs,
                                "kv_size": f"{kv_size} / {full_seq_len}",  
                                "compress": compress_count,
                                "final_compress": final_compress, 
                                "time(sec)": total_time, 
                                "avg_chunk": avg_chunking, 
                                "avg_compress": avg_compress, 
                                "label": item['completion']
                                }, ensure_ascii=False)+'\n')

f.close()
print("Done")
