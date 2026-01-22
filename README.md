# Synaptic Caching : 임계값 기반, 작업 적응형 KV 캐시 관리 및 효율적이고 일관된 장기 컨텍스트 생성

## 개요

기존 Transformer 모델은 입력 길이가 제한적이라는 구조적 한계가 있습니다. 이 프로젝트는 이러한 한계를 넘어, 길이가 긴 입력(Long Context)을 잘림이나 손실 없이 온전하게 처리하기 위한 새로운 접근 방식을 제안하고 검증합니다.

- Streaming LLM(https://github.com/mit-han-lab/streaming-llm)의 코드를 일부 차용함
### 제안 방법
입력 데이터를 청크(Chunk) 단위로 나누어 처리함으로써 연산 효율을 확보하고, 메모리가 캐시 사이즈(Cache Size) 한계에 도달하면 압축 알고리즘을 가동합니다. 이때 어떤 토큰을 남길지 결정하는 핵심 지표로 내부 어텐션 스코어를 활용하며, 일시적인 중요도 변화나 노이즈를 걸러내기 위해 Top-K, EMA, Gaussian 등의 통계적 기법을 적용합니다. 결과적으로 중요도가 검증된 토큰들만 캐시에 남아 다음 토큰 생성 시 활용되므로, 모델은 제한된 리소스 안에서도 긴 문맥의 의미를 잃지 않고 고품질의 생성을 이어갈 수 있습니다.
### 환경설정

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
``` 

### 실행 방법
#### LongBench 
```bash
CUDA_VISIBLE_DEVICES=3 python examples/longbench_test.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --enable_start_recent_kv_cache --start_size 160 --recent_use yes --compress gaus --chunk_size 700 --cache_size 7000 --trigger_size 6180
```

#### Booksum
 ```bash
 CUDA_VISIBLE_DEVICES=3 python examples/booksum_test.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --enable_start_recent_kv_cache --start_size 160 --recent_size 1336 --dataset_name data/booksum.jsonl --recent_use no --compress topk --chunk_size 700 --cache_size 7000 --trigger_size 6180
 ```
 - dataset name : 데이터 path (Booksum에만 사용)
 - chunk size : 인코딩 시 한번에 집어 넣을 토큰의 길이
 - cache size : K/V 캐시의 목표 규모
 - trigger size : K/V 길이가 압축되는 임계값
 - compress : 압축 알고리즘(gaus/ema/topk)
 - recent use : Recent Window 사용 여부(yes/no)
 - start size : Sink 크기



