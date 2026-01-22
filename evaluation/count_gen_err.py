import json
import sys
import transformers 

tok = transformers.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

filename = sys.argv[1]
with open(filename) as f :
    data = [json.loads(d) for d in f]

err = 0
for line in data :
    pred = tok(line['pred'])['input_ids']
    longest = 0
    label = ""
    
    for i in line['label']:
        length = tok(i)['input_ids']
        if len(length) >= longest :
            longest = len(length)
            label = i
    
    if len(label) < len(pred) and len(pred) >= 500 :
        err += 1

print(f"Err / Total: {err} / {len(data)}")
