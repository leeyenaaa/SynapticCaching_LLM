import torch
import transformers

device = 'cuda'
sentence = "I'm going to make a chocolate chip cookie today. Could you write a recipe to make it that have accurace weighning? By the way, I don't have a measuring cup, I only have a regular paper cup."

model = transformers.AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct').to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt').to(device)
position_ids = torch.arange(0, input_ids.shape[-1], device=device).unsqueeze(0)
exclude_indices = [5, 9, 10, 11, 20, 21, 26, 31]

mask = torch.ones(input_ids.size(1), dtype=torch.bool)
mask[exclude_indices] = False
input_ids = input_ids[:, mask]
position_ids = position_ids[:, mask]

seq_len = input_ids.shape[-1]
#output = model.generate(input_ids) #position_ids=position_ids_tmp)
#print(tokenizer.decode(output[0]))

past_key_values = None
gen_result = list()
pos = 0
result = list()
for _ in range(1000-1) :
    outputs = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    position_ids = torch.cat([position_ids, torch.tensor([[position_ids.shape[-1].item()]])], dim=-1)

    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    gen_result.append(pred_token_idx.item())

    generated_text = (
            tokenizer.decode(gen_result, skip_special_tokens=True, 
                clean_up_tokenization_spaces=True, spaces_between_special_tokens=False).strip().split(" "))
    now = len(generated_text)-1
    if now>pos:
        result.append(generated_text[pos:now][-1])
        pos=now
    if pred_token_idx == 128009: break


print(" ".join(result))
