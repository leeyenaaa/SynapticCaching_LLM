## 예제 실행 커맨드 
``` shell
CUDA_VISIBLE_DEVICES=3 python longbench_streaming_w_EMA_RoPE_regen.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \\
					 --enable_start_recent_kv_cache --start_size 160 --recent_use no --compress gaus
``` 


## 문제가 된 부분 

### 1) KV용 Position 전달 
- huggingface api로 전달할 수 있는 `position_ids`는 input sequence를 위한 `position_id`이며, `past_key_values`를 위한`position_id`는 없었다. 
- 해당 문제에 대하여 `position_ids`에 `past_key_values`용 id 와 `input_sequence`용 id를 concatenation하여 전달 
  - `longbench_streaming_w_EMA_RoPE_regen.py` 중 ⭐️121, 158, 178번 라인 참고 
- concat된 `position_ids`는 `streaming_llm/pos_shift/modify_llama_shift.py`에서 분할되어 각 역할에 맞게 RoPE에 참가함 
  - `streaming_llm/pos_shift/modify_llama_shift.py` 중 ⭐️86번 라인 참고 


### 2) Decoding 진입 시 Position ID와 Seq length 간 문제 
- huggingface api 내에서 배치 연산을 가정하였기 때문에 `position_ids`의 개수와 `seq_length`가 배수 관계여야 하는 문제가 있음
- 해당 문제에 대하여 `position_ids`를 `seq_length`의 배수이고 `position_ids_len`보다 크면서 가장 가까운 수만큼 무의미한 값으로 채우고 이후 삭제한다. 
	- ex) `seq_length`가 100, `position_ids_len`가 2950인 경우 `position_ids`의 길이가 3000이 될 때까지 -999로 채운다. 
	- `longbench_streaming_w_EMA_RoPE_regen.py` 중 ⭐️160번 라인 참고 

- 삭제 시에는 `past_key_values`의 길이에 맞춰 `position_ids`를 잘라낸다. 
	- `streaming_llm/pos_shift/modify_llama_shift.py` 중 ⭐️89번 라인 참고 
