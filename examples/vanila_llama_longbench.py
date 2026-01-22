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

templates = json.load(open("streaming_llm/dataset2prompt.json", 'r'))

model, tokenizer = load(args.model_name_or_path)
max_gen_len = 1000 
max_context_len = model.config.max_position_embeddings

output_filepath = f"results/LongBench/test/test_vanila_llama3"
print(f"Saving to... \n>>> {output_filepath}\nConfirm(y/n) >>>", end='')
ans = input()
if ans != 'y' : exit()

os.makedirs(output_filepath, exist_ok=True)

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
datasets = ["2wikimqa"]
print(datasets)
data = dict()
for name in datasets:
    temp = load_dataset('zai-org/LongBench', name, split='test')
    data[name] = [x for x in temp]



os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")
num_eval_tokens = 0 

max_length = model.config.max_position_embeddings

for name in datasets:
    with open(f"{output_filepath}/{name}.jsonl", 'w') as f :
        for i, item in enumerate(tqdm(data[name])):
            result = list()
            prompt = templates[name]
            inputs = prompt.format(**item)
            encodings = tokenizer.encode(inputs, add_special_tokens=True, return_tensors='pt').to(device)

            flg = 0
        # === Vanila Llama3 === 
            if encodings.shape[-1] > max_length :
                available = encodings.shape[-1] - 8000
                inputs = encodings[:, available:]

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

                
            for_save = {
                    "input": item['input'], "pred": output_sentence, 'label': item['answers'], 
                    'length': item['length'], 'dataset': item['dataset'], 'language': item['language'], 
                    'all_classes': item['all_classes'], '_id': item['_id']}

            f.write(json.dumps(for_save, ensure_ascii=False)+'\n')


f.close()
print("Done")
