import torch
import json
import math
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# enable_streaming_llm 및 관련 유틸리티가 올바르게 임포트되었다고 가정.
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.utils import parse_args, load
# modify_llama_shift가 다른 곳에서 적용되었거나 enable_streaming_llm에 의해 처리된다고 가정.
# from streaming_llm.pos_shift.modify_llama_shift import enable_llama_pos_shift_attention

device = 'cuda'
args = parse_args()

templates = json.load(open("streaming_llm/dataset2prompt.json", 'r'))
compress_map = json.load(open("streaming_llm/dataset2compress.json", 'r')) # 변수명 변경

model, tokenizer = load(args.model_name_or_path)

# --- 설정 ---
max_gen_len = 1000
max_context_len = model.config.max_position_embeddings # 또는 args.max_context_len (정의된 경우)
recent_use = args.recent_use
chunk_size = 819 # 예시 값
cache_size = 6144 # 예시 값
trigger_size = 5324 # 예시 값
k_count_encoding = chunk_size * 4 # 명확성을 위해 변수명 변경
k_count_decoding = chunk_size * 6 # 명확성을 위해 변수명 변경

output_filepath = f"results/LongBench/RoPE/test_streaming_{args.compress}_chunk{chunk_size}_cache{cache_size}_trig{trigger_size}_kE{k_count_encoding}_kD{k_count_decoding}_recent{args.recent_use}"
print(f"설정 확인:")
print(f"  Chunk Size: {chunk_size}")
print(f"  Cache Size (Budget): {cache_size}")
print(f"  Trigger Size: {trigger_size}")
print(f"  K Encoding: {k_count_encoding}")
print(f"  K Decoding: {k_count_decoding}")
print(f"  Recent Use: {args.recent_use}")
print(f"출력 파일 경로: {output_filepath}")
ans = input("(y/n) >>> ")
if ans.lower() != 'y':
    exit()

os.makedirs(output_filepath, exist_ok=True)

# 처리할 데이터셋 정의
# datasets = ["narrativeqa", "qasper", ...] # 전체 목록
datasets = [ "triviaqa", "musique", "hotpotqa", "2wikimqa", "qasper", 'samsum', 'dureader', 'lsht', 'gov_report'] # 예시 부분집합

# 데이터셋 로드
data = dict()
for name in datasets:
    temp = load_dataset('zai-org/LongBench', name, split='test')
    data[name] = [x for x in temp]

# args.compress에 따라 데이터셋별 압축 방법 결정
compression_methods = {}
if args.compress == 'gaus':
    default_comp = 'gaussian'
elif args.compress == 'ema':
    default_comp = 'moving_average'
elif args.compress == 'peak':
    default_comp = 'peak_finding'
else:
    default_comp = 'top_k'

for name in datasets:
    # compress_map에 특정 방법이 있으면 사용, 없으면 기본값 사용
    compression_methods[name] = compress_map.get(name, default_comp)
print(f"사용 압축 방법: {compression_methods}")

# StreamingLLM 로직 및 필요한 경우 사용자 정의 어텐션 활성화
attn_score_placeholder = None # 축출 로직에 어텐션 점수가 필요한 경우를 위한 플레이스홀더
if args.enable_start_recent_kv_cache:
    # enable_streaming_llm이 객체를 반환하거나 모델을 수정하고,
    # 필요한 경우 캐시 관리자 객체를 반환한다고 가정.
    # `evict_for_space_analysis` 메서드를 가진 캐시 관리자 `kv_cache_manager`가 필요.
    kv_cache_manager = enable_streaming_llm(
        model, recent_use=recent_use, cache_size=cache_size,
        start_size=args.start_size, recent_size=args.recent_size, compress=args.compress
    )
    # 필요한 경우 enable_llama_pos_shift_attention(model) 호출 확인
else:
    kv_cache_manager = None # 축출이 발생하지 않음

# 로그 파일 설정
os.makedirs(args.output_dir, exist_ok=True)
# f_log = open(f"{args.output_dir}/log.txt", "w") # 스니펫에서는 사용되지 않음, 다른 곳에서 필요하면 유지

# --- 메인 처리 루프 ---
for name in datasets:
    # 재개를 허용하기 위해 기존 결과 확인
    output_file = f"{output_filepath}/{name}.jsonl"
    exist_len = 0
#    if os.path.exists(output_file):
#        try:
#            with open(output_file, "r") as f_in:
#                exist_len = sum(1 for _ in f_in)
#        except Exception as e:
#            print(f"경고: 기존 파일 {output_file}을 읽을 수 없습니다. 오류: {e}")
#            exist_len = 0
#
#    if exist_len >= len(data[name]):
#        print(f"데이터셋 {name} 건너뛰기, 이미 완료됨 ({exist_len} 항목).")
#        continue
#    else:
#         print(f"데이터셋 {name} 처리 중, 항목 {exist_len}부터 시작.")

    current_compression_method = compression_methods[name]

    with open(output_file, 'w') as f_out:
        for i, item in enumerate(tqdm(data[name][2:], desc=f"Processing {name}")):
            generated_output_sentence = "" # 각 항목에 대해 초기화
            error_occurred = False # 오류 플래그

            try:
                # 프롬프트 준비
                prompt_template = templates[name]
                inputs_text = prompt_template.format(**item)
                input_ids = tokenizer.encode(inputs_text, add_special_tokens=True, return_tensors='pt').to(device)
                seq_len_total = input_ids.shape[-1]

                # --- 인코딩 단계 ---
                past_key_values = None   # KV 캐시
                past_positions = None    # KV 캐시의 절대 위치 ID
                attn_scores_for_eviction = None # 축출 로직에 필요한 최신 어텐션 점수 저장

                input_chunks = input_ids.split(chunk_size, dim=1)
                num_chunks = len(input_chunks)
                attn_flg = False 

                for chunk_idx, current_chunk in enumerate(input_chunks):
                    if chunk_idx == num_chunks - 1: continue # 마지막 청크는 인코딩 단계에서 제외

                    current_chunk_len = current_chunk.shape[-1]
                    current_kv_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]

                    # --- 필요 시 캐시 확인 및 축출 ---
                    if kv_cache_manager is not None and current_kv_len >= trigger_size:
                        attn_flg = True 
                        if attn_scores_for_eviction is None:
                            print("경고: 트리거 크기에 도달했지만 축출에 사용할 어텐션 점수가 없습니다.")
                        else:
                            try:
                                # **중요**: 이 함수가 업데이트된 pkv와 업데이트된 positions을 모두 반환하는지 확인
                                past_key_values, past_positions = kv_cache_manager.evict_for_space_analysis(
                                    past_key_values, attn_scores_for_eviction, current_compression_method,
                                    k_count_encoding, past_positions
                                )
                                # **검증**: 축출 후 길이가 일치하는지 확인
                                assert past_key_values[0][0].shape[-2] == past_positions.shape[-1], \
                                    f"인코딩 축출 후 불일치! KV 길이: {past_key_values[0][0].shape[-2]}, Pos 길이: {past_positions.shape[-1]}"
                                current_kv_len = past_key_values[0][0].shape[-2] # 축출 후 길이 업데이트
                            except Exception as e:
                                print(f"청크 {chunk_idx}에서 인코딩 축출 중 오류 발생: {e}")
                                error_occurred = True
                                break
                    # --- Position ID 준비 ---
                    start_pos = 0 if past_positions is None else past_positions[0, -1].item() + 1
                    current_pos_ids = torch.arange(start_pos, start_pos + current_chunk_len, device=device).unsqueeze(0) # Shape: [1, current_chunk_len]

                    # 어텐션 모듈을 위해 과거 위치와 현재 위치 연결
                    position_ids_for_model = current_pos_ids if past_positions is None else torch.cat([past_positions, current_pos_ids], dim=-1)

                    # --- 순전파 (인코딩) ---
                    with torch.no_grad():
                        outputs = model(
                            input_ids=current_chunk,
                            past_key_values=past_key_values,
                            position_ids=position_ids_for_model,
                            output_attentions=True, # 축출 점수 계산을 위해 어텐션 필요
                            return_dict=True,
                            use_cache=True
                        )
#                    if attn_flg :
#                        exit()
                    # --- 상태 업데이트 ---
                    past_key_values = outputs.past_key_values
                    # 성공적인 순전파 후에만 past_positions 업데이트
                    past_positions = current_pos_ids if past_positions is None else torch.cat([past_positions, current_pos_ids], dim=1)
                    attn_scores_for_eviction = list(outputs.attentions) # 최신 어텐션 저장
#                    if attn_flg:
#                        from streaming_llm.visualize import visualize_selected_attentions
#                        visualize_selected_attentions(outputs.attentions)
#                        

                #sentence = "I'm going to make a chocolate chip cookie today. Could you write a recipe to make it that have accurace weighning? By the way, I don't have a measuring cup, I only have a regular paper cup."
                #sentence = "Summarize following sentence: The next day, while still in bed, the doctor’s wife said to her husband, We have little food left, we’ll have to go out again, I thought that today I would go back to the underground food store at the supermarket, the one I went to on the first day, if nobody else has found it, we can get supplies for a week or two, I’m coming with you and we’ll ask one or two of the others to come along as well, I’d rather go with you alone, it’s easier, and there is less danger of getting lost, How long will you be able to carry the burden of six helpless people, I’ll manage as long as I can, but you are quite right, I’m beginning to get exhausted, sometimes I even wish I were blind as well, to be the same as the others, to have no more obligations than they have, We’ve got used to depending on you, If you weren’t there, it would be like being struck with a second blindness, thanks to your eyes we are a little less blind, I’ll carry on as long as I can, I can’t promise you more than that, One day, when we realize that we can no longer do anything good and useful we ought to have the courage simply to leave this world, as he said, Who said that, The fortunate man we met yesterday, I am sure that he wouldn’t say that today, there is nothing like real hope to change one’s opinions, He has that all right, long may it last, In your voice there is a tone which makes me think you are upset, Upset, why, As if something had been taken away from you, Are you referring to what happened to the girl when we were at that terrible place, Yes, Remember it was she who wanted to have sex with me, Memory is deceiving you, you wanted her, Are you sure, I was not blind, Well, I would have sworn that, You would only perjure yourself, Strange how memory can deceive us, In this case it is easy to see, something that is offered to us is more ours than something we had to conquer, But she didn’t ever approach me again, and I never approached her, If you wanted to, you could find each other’s memories, that’s what memory is for, You are jealous, No, I’m not jealous, I was not even jealous on that occasion, I felt sorry for her and for you, and also for myself because I could not help you, How are we fixed for water, Badly."
#                sentence = "Please summarize the following passage in 3–4 concise sentences, focusing only on the main ideas and avoiding unnecessary details. \n\nText: In recent years, climate change has accelerated due to rising greenhouse gas emissions. This has led to unpredictable weather patterns, severe droughts, and record-breaking heatwaves across many continents. Governments are struggling to implement effective climate policies, while industries face increasing pressure to adopt cleaner technologies. Meanwhile, individuals are becoming more aware of their carbon footprint and demanding stronger global cooperation."
#
#                input_ids = tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt').to(device)
#                last_chunk_len = input_ids.shape[-1]
#                position_ids = torch.arange(0, input_ids.shape[-1], device=device).unsqueeze(0)
#                rand_indices = torch.randperm(last_chunk_len)[:int(last_chunk_len*0.5)]
#                
#                print(tokenizer.decode(input_ids[:, rand_indices][0]))
#
#                mask = torch.ones(input_ids.size(1), dtype=torch.bool)
#                mask[rand_indices] = False
#                input_ids = input_ids[:, mask]
#                position_ids = position_ids[:, mask]
#                past_key_values = None 
#                past_positions = None
#                last_chunk = input_ids
#                last_chunk_len = input_ids.shape[-1]
                if error_occurred: # 인코딩 실패 시 디코딩 건너뛰기
                    generated_output_sentence = f"인코딩 단계 청크 {chunk_idx}에서 오류 발생."

                # --- 디코딩 단계 ---
                else:
                    last_chunk = input_chunks[-1]
                    last_chunk_len = last_chunk.shape[-1]
                    current_kv_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]

                    # --- 첫 디코딩 단계를 위한 Position ID 준비 ---
                    start_pos = 0 if past_positions is None else past_positions[0, -1].item() + 1
                    current_pos_ids = torch.arange(start_pos, start_pos + last_chunk_len, device=device).unsqueeze(0)
                    position_ids_for_model = current_pos_ids if past_positions is None else torch.cat([past_positions, current_pos_ids], dim=-1)

                    # --- Hugging Face API 제약 조건 해결을 위한 워크어라운드 ---
                    # API는 `position_ids` 길이가 `input_ids` 길이와 특정 관계를 갖기를 기대할 수 있슴.
                    # 특히 past_key_values가 있을 때 내부 리셰이핑 등을 위해 배수 또는 특정 차원과 일치하도록 패딩이 필요할 수 있슴.
                    # 이 패딩은 수정된 어텐션 내부에서 제거.
                    effective_input_len = last_chunk_len # 이 단계의 실제 입력 길이
                    required_len_multiple = effective_input_len
                    current_pos_len = position_ids_for_model.shape[-1]
                    padding_needed = (math.ceil(current_pos_len / required_len_multiple) * required_len_multiple) - current_pos_len
                    if padding_needed > 0:
                        # 어텐션 내부에서 확인되는 유효하지 않은 토큰 ID 인덱스 값 사용
                        padding_tensor = torch.full((1, padding_needed), -999, device=device, dtype=torch.long)
                        position_ids_for_model = torch.cat([position_ids_for_model, padding_tensor], dim=-1)
                        # print(f"DEBUG: 입력 길이 {effective_input_len}에 대해 position_ids를 {current_pos_len}에서 {position_ids_for_model.shape[-1]}로 패딩함")
                    # --- 워크어라운드 종료 ---

                    generated_ids = []
                    generated_text_tokens = [] # 쉬운 결합을 위해 생성된 토큰 저장
                    current_generated_pos = 0 # 텍스트 위치 추적

                    with torch.no_grad():
                        # --- 첫 디코딩 단계 (마지막 입력 청크 처리) ---
                        try:
                             outputs = model(
                                input_ids=last_chunk,
                                past_key_values=past_key_values,
                                position_ids=position_ids_for_model,
                                output_attentions=True,
                                use_cache=True
                            )
                        except torch.cuda.OutOfMemoryError:
                            print(f"첫 디코딩 단계 중 OOM 오류. 축출 시도.")
                            if kv_cache_manager is not None and attn_scores_for_eviction is not None:
                                past_key_values, past_positions = kv_cache_manager.evict_for_space_analysis(
                                    past_key_values, attn_scores_for_eviction, current_compression_method,
                                    k_count_decoding, past_positions # 디코딩 K 값 사용
                                )
                                assert past_key_values[0][0].shape[-2] == past_positions.shape[-1], \
                                    f"OOM 축출 후 불일치! KV 길이: {past_key_values[0][0].shape[-2]}, Pos 길이: {past_positions.shape[-1]}"
                                # 재시도를 위해 position ID 재계산
                                start_pos = 0 if past_positions is None else past_positions[0, -1].item() + 1
                                current_pos_ids = torch.arange(start_pos, start_pos + last_chunk_len, device=device).unsqueeze(0)
                                position_ids_for_model = current_pos_ids if past_positions is None else torch.cat([past_positions, current_pos_ids], dim=-1)
                                # 필요한 경우 패딩 다시 적용
                                current_pos_len = position_ids_for_model.shape[-1]
                                padding_needed = (math.ceil(current_pos_len / required_len_multiple) * required_len_multiple) - current_pos_len
                                if padding_needed > 0:
                                     padding_tensor = torch.full((1, padding_needed), -999, device=device, dtype=torch.long)
                                     position_ids_for_model = torch.cat([position_ids_for_model, padding_tensor], dim=-1)
                                # 순전파 재시도
                                outputs = model(
                                    input_ids=last_chunk, past_key_values=past_key_values,
                                    position_ids=position_ids_for_model, output_attentions=True, use_cache=True
                                )
                            else:
                                raise # 축출 불가능 시 OOM 다시 발생시킴


                        past_key_values = outputs.past_key_values
                        past_positions = current_pos_ids if past_positions is None else torch.cat([past_positions, current_pos_ids], dim=1)
                        attn_scores_for_eviction = list(outputs.attentions)
                        if attn_flg :
                            from streaming_llm.visualize import visualize_selected_attentions
                            visualize_selected_attentions(outputs.attentions)
                        
                        # 첫 예측 토큰 얻기
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        generated_ids.append(next_token_id.item())

                        # --- 후속 디코딩 단계 (토큰 단위) ---
                        for step in range(max_gen_len - 1):
                            current_kv_len = past_key_values[0][0].shape[-2]

                            # --- 필요 시 캐시 확인 및 축출 ---
                            if kv_cache_manager is not None and current_kv_len >= trigger_size:
                                if attn_scores_for_eviction is None:
                                     print("경고: 디코딩 중 트리거 크기에 도달했지만 어텐션 점수가 없습니다.")
                                else:
                                    try:
                                        past_key_values, past_positions = kv_cache_manager.evict_for_space_analysis(
                                            past_key_values, attn_scores_for_eviction, current_compression_method,
                                            k_count_decoding, past_positions
                                        )
                                        # **검증**
                                        assert past_key_values[0][0].shape[-2] == past_positions.shape[-1], \
                                            f"디코딩 축출 후 불일치! KV 길이: {past_key_values[0][0].shape[-2]}, Pos 길이: {past_positions.shape[-1]}"
                                    except Exception as e:
                                        print(f"단계 {step}에서 디코딩 축출 중 오류 발생: {e}")
                                        error_occurred = True
                                        break


                            # --- 이 단계를 위한 Position ID 준비 ---
                            start_pos = past_positions[0, -1].item() + 1
                            current_pos_ids = torch.arange(start_pos, start_pos + next_token_id.shape[-1], device=device).unsqueeze(0)
                            position_ids_for_model = torch.cat([past_positions, current_pos_ids], dim=-1)
                            # 여기서는 입력이 단일 토큰(effective_input_len=1)이므로 패딩 필요 없음

                            # --- 순전파 (디코딩 단계) ---
                            outputs = model(
                                input_ids=next_token_id,
                                past_key_values=past_key_values,
                                position_ids=position_ids_for_model,
                                output_attentions=True,
                                use_cache=True
                            )
                            input("confirm: ")
                            # --- 상태 업데이트 ---
                            past_key_values = outputs.past_key_values
                            past_positions = torch.cat([past_positions, current_pos_ids], dim=1)
                            attn_scores_for_eviction = list(outputs.attentions)

                            # 다음 예측 토큰 얻기
                            next_token_logits = outputs.logits[:, -1, :]
                            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                            generated_ids.append(next_token_id.item())

                            # --- EOS 확인 ---
                            # 사용 가능한 경우 토크나이저의 EOS ID 사용, 그렇지 않으면 일반적인 ID 사용. Llama 3는 128009 사용
                            if next_token_id.item() == tokenizer.eos_token_id or next_token_id.item() == 128009:
                                print(f"EOS token {next_token_id.item()} detected at step {step}. Stopping generation.") # EOS 로그 추가
                                break
#                        else: # for 루프가 break 없이 완료된 경우
#                             print(f"Warning: Max generation length {max_gen_len} reached without EOS token.")


                    # 최종 생성된 시퀀스 디코딩
                    generated_output_sentence = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    # 생성된 문장 로깅 (디버깅 목적)
                    # print(f"Dataset: {name}, Item: {i+exist_len}, Generated: '{generated_output_sentence[:100]}...'")

            except Exception as e:
                 print(f"데이터셋 {name}의 항목 {i+exist_len} 처리 중 오류 발생: {e}")
                 generated_output_sentence = f"오류: {e}"
                 error_occurred = True # 오류 플래그 설정 확인

            # 결과 저장 (오류 발생 시에도)
            final_kv_len = "N/A" if past_key_values is None else past_key_values[0][0].shape[-2]
            for_save = {
                "input": item.get('input', 'N/A'),
                "pred": generated_output_sentence,
                'label': item.get('answers', 'N/A'),
                "kv_size_at_end": str(final_kv_len),
                'length': item.get('length', 'N/A'),
                'dataset': item.get('dataset', name),
                'language': item.get('language', 'N/A'),
                'all_classes': item.get('all_classes', 'N/A'),
                '_id': item.get('_id', f"{name}_{i+exist_len}"),
                'error_occurred': error_occurred
            }
            f_out.write(json.dumps(for_save, ensure_ascii=False) + '\n')

# 사용한 경우 로그 파일 닫기
# f_log.close()
print(f"처리 완료. 결과가 {output_filepath}에 저장되었습니다.")
