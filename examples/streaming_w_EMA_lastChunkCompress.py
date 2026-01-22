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
chunk_size = int(max_context_len * 0.15)
cache_size = int(max_context_len * 0.440)
trigger_size = int(max_context_len * 0.350)
k_count = chunk_size

print(f"Confirm: {chunk_size}, {cache_size}, {trigger_size} >>>", end='')
attn_score = None
if args.enable_start_recent_kv_cache: 
    kv_cache = enable_streaming_llm(
            model, recent_use=recent_use, cache_size=cache_size, start_size=args.start_size, recent_size=args.recent_size)
else:
    kv_cache = None

if args.enable_pos_shift:
   from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention


#output_filepath = f"results/booksum/streaming_gaus_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_recent{args.recent_use}_lastChunkCompress.jsonl"
output_filepath = "results/booksum/compress_test.jsonl"
os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

print(f"Saving to... \n>>> {output_filepath}\nConfirm(y/n) >>>", end='')
ans = input()
if ans != 'y':
    exit()

with open(f"{output_filepath}", 'w') as f :
    for i, item in enumerate(tqdm(data)):
        result = list() 
        inputs = item['prompt'].replace('\n\nQ: Can you write an appropriate summary of the above paragraphs?\nA:', '')
        inputs = item['prompt'].replace('Chapter:', '')
        inputs = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful chat bot that summarizes books.<|eot_id|><|start_header_id|>user<|end_header_id|>f'Summarize the following text in about 300 words:\n{inputs}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        encodings = tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt').to(device)
        past_key_values = None 
        # If Overflow: Chunking + Encoding w/RecentKV, EMA 
        chunks = encodings.split(chunk_size, dim=1)

        for chunk in range(len(chunks)) :
            if chunk == len(chunks)-1: continue
            if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:
                # (7600/8) * 6650? 
                # Pruning with about EMA score
                past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, k_count)
                #print("Compress")

            with torch.no_grad():
                try :
                    c_output = model(input_ids=chunks[chunk].to(device), past_key_values=past_key_values, output_attentions=True, use_cache=True)
                
                except:
                    print(chunks[chunk].shape)
                    print(past_key_values[0][0].shape)
                    result.append(f"Memory Out Err in encoding process: {encodings.shape} / {past_key_values[0][0].shape}")
                    break
                past_key_values = c_output.past_key_values
                attn_score = list(c_output.attentions)
            #print(past_key_values[0][0].shape)

        #encodings = chunks[-1]
        
        if past_key_values is not None and past_key_values[0][0].shape[-2] >= trigger_size:
            # (7600/8) * 6650? 
            # Pruning with about EMA score
            past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, k_count)
            #print("Compress")

        #print(past_key_values[0][0].shape)        
        seq_len = encodings.shape[1]
        
#        with torch.no_grad() :
#            try: outputs = model(input_ids=encodings, past_key_values=past_key_values, use_cache=True)
#            except:
#                print(encodings.shape)
#                print(past_key_values[0][0].shape)
#                result.append("Memory Out Err: {encodings.shape} / {past_key_values[0][0].shape}")
#                break
#            past_key_values = outputs.past_key_values
#            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
#            generated_ids = [pred_token_idx.item()]
#            pos = 0
#            for _ in range(max_gen_len - 1):
#                outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, output_attentions=True, use_cache=True)
#                past_key_values = outputs.past_key_values 
#                attn_score = list(outputs.attentions)
#
#                if kv_cache is not None:
#                    past_key_values = kv_cache.evict_for_space_analysis(past_key_values, attn_score, k_count)
#                
#                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
#                generated_ids.append(pred_token_idx.item())
#                generated_text = (
#                        tokenizer.decode(generated_ids, skip_special_tokens=True, 
#                            clean_up_tokenization_spaces=True, spaces_between_special_tokens=False).strip().split(" "))
#                now = len(generated_text) - 1
#                if now > pos :
#                    #print(generated_text[pos:now][-1], end=" ")
#                    result.append(generated_text[pos:now][-1])
#                    pos = now
#                if pred_token_idx == 128009:
#                    break
        

#        result.append(generated_text[pos:][-1])
#        output_sentence = " ".join(result)
        if past_key_values is None :
            kv_len = 0 
        else :
            kv_len = past_key_values[0][0].shape

        f.write(json.dumps({
                                #"input": inputs,
                                "past_kv": f"{kv_len} / {seq_len}", 
                                "output": "",  #output_sentence, 
                                "label": item['completion']
                                }, ensure_ascii=False)+'\n')

f.close()
print("Done")
