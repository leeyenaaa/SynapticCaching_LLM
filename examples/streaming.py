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
if ".jsonl" not in args.dataset_name:
    data = load_dataset(args.dataset_name, split=args.split)
else :
    with open(args.dataset_name, 'r') as f:
        data = [json.loads(d) for d in f]
    
model, tokenizer = load(args.model_name_or_path)
max_gen_len = 1000 

past_key_values = None
if args.enable_start_recent_kv_cache: 
    kv_cache = enable_streaming_llm(
            model, cache_size=args.start_size + args.recent_size, start_size=args.start_size, recent_size=args.recent_size, 
            recent_use=args.recent_use)
else:
    kv_cache = None

if args.enable_pos_shift:
   from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

output_filepath = f"results/booksum/test_streaming_llm.jsonl"

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

print(f"Saving to... \n>>> {output_filepath}\nConfirm(y/n) >>>", end='')
ans = input()
if ans != 'y':
    exit()

max_length = model.config.max_position_embeddings

with open(output_filepath, 'a') as f :
    for i, item in enumerate(tqdm(data)):
        flg = 0
        inputs = item['prompt']
        prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful chat bot that summarizes books.<|eot_id|><|start_header_id|>user<|end_header_id|>Summarize the following text in about 300 words:"
        suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        prompt = "".join([prefix, inputs, suffix])
        encodings = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
        
        # truncate in middle (ref: StreamingLLM Paper Appendix D) 
        if encodings.shape[-1] > 7600 :
            half = int(7600 / 2) # max length of Llama3 = 8192 ==> 8100 ==> 8100 - 500 = 7600 
            prompt = tokenizer.decode(encodings[:, :half][0], skip_special_tokens=False)+tokenizer.decode(encodings[:, -half:][0], skip_special_tokens=False)
            encodings = tokenizer(prompt, truncation=False, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        

        seq_len = encodings.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed) 
            #past_key_values = kv_cache(past_key_values) 
        
        with torch.no_grad() :
            try :
                outputs = model(input_ids=encodings, past_key_values=past_key_values, use_cache=True)
            except torch.OutOfMemoryError as e:
                flg = 1 
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids = [pred_token_idx.item()]
            pos = 0
            result = list() 
            for _ in range(max_gen_len - 1):
                if flg == 1 :
                    output_sentence = "Memory Out Err in Encoding Process"                    
                    break

                try :
                    outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
                except Exception as e :
                    flg = 1 
                    output_sentence = "Memory Out Err in Decoding Process"
                    break

                past_key_values = outputs.past_key_values 
                space_needed = 1
                if kv_cache is not None :
                    #past_key_values = kv_cache(past_key_values)
                    past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_ids.append(pred_token_idx.item())
                generated_text = (
                        tokenizer.decode(generated_ids, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=True, spaces_between_special_tokens=False).strip().split(" "))
                now = len(generated_text) - 1
                if now > pos :
                    #print(generated_text[pos:now][-1], end=" ")
                    result.append(generated_text[pos:now][-1])
                    pos = now
                if pred_token_idx == 128009:
                    break
        
        if flg != 1 :
            result.append(generated_text[pos:][-1])
            output_sentence = " ".join(result)
         
        f.write(json.dumps({
                                #"input": inputs, 
                                "output": output_sentence, 
                                "label": item['completion']
                                }, ensure_ascii=False)+'\n')

f.close()
print("Done")
