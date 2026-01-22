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

output_filepath = f"results/booksum/test_vanila_llama3.jsonl"
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
    # === Vanila Llama3 === 
        if encodings.shape[-1] > max_length :
            prefix = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt')
            inputs = tokenizer.encode(inputs, add_special_tokens=False, return_tensors='pt')
            suffix = tokenizer.encode(suffix, add_special_tokens=False, return_tensors='pt')

            available = 8000 - prefix.shape[-1] - suffix.shape[-1]
            inputs = inputs[:, -available:]
            encodings = torch.cat([prefix, inputs, suffix], dim=-1).to(device)

        seq_len = encodings.shape[1]
        
        with torch.no_grad() :
            try :
                outputs = model.generate(input_ids=encodings, max_new_tokens=max_gen_len, pad_token_id=tokenizer.eos_token_id)
            except torch.OutOfMemoryError as e:
                flg = 1  
            
        if flg == 1 :
            output_sentence = "Memory Out Err in Encoding Process"                    
        else :
            output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            inputs = tokenizer.decode(encodings[0], skip_special_tokens=True)
            output_sentence = output_sentence.replace(inputs, "")

            
         
        f.write(json.dumps({
                                #"input": inputs, 
                                "output": output_sentence, 
                                "label": item['completion']
                                }, ensure_ascii=False)+'\n')

f.close()
print("Done")
