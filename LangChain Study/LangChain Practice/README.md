# LangChain 실습

## 목차

* [1. 기본 요구사항](#1-기본-요구사항)
* [2. 개발 일정](#2-개발-일정)
* [3. 구현 내용 상세](#3-구현-내용-상세)
  * [3-1. 학습 데이터셋 생성 과정](#3-1-학습-데이터셋-생성-과정) 
* [4. LLM 선택](#4-llm-선택)
  * [4-1. 한국어 LLM 선택 이유](#4-1-한국어-llm-선택-이유)
  * [4-2. 한국어 LLM 성능 상세 비교 결과](#4-2-한국어-llm-성능-상세-비교-결과)
* [5. 이슈 사항 및 해결 방법](#5-이슈-사항-및-해결-방법)
  * [5-1. EOS token 학습 안됨](#5-1-eos-token-학습-안됨)
  * [5-2. LLM Fine-Tuning 후, 응답이 제대로 생성되지 않음](#5-2-llm-fine-tuning-후-응답이-제대로-생성되지-않음)
  * [5-3. LLM output 에서 처음에 EOS token 발생](#5-3-llm-output-에서-처음에-eos-token-발생)
  * [5-4. Fine-Tuning 된 LLM 로딩 시 tensor size 불일치](#5-4-fine-tuning-된-llm-로딩-시-tensor-size-불일치)
* [6. 참고](#6-참고)
  * [6-1. Quantization 적용/미 적용 시 GPU 메모리 사용량 차이](#6-1-quantization-적용미-적용-시-gpu-메모리-사용량-차이) 

## 1. 기본 요구사항

* 다음 tool call 을 하는 간단한 LLM 에이전트 개발
  * 특정 날짜로부터 ```N```일 전/후의 날짜 계산
  * 특정 날짜의 요일 계산 및 반환
* 구현 필수 사항
  * Open Source LLM 을 ```trl``` 라이브러리를 이용하여 Fine-Tuning  
  * LangChain 기반으로 구현
  * [LangChain 에이전트 tool call 문서](../LangChain_에이전트_tool_call.md) 를 참고하여 tool call 방식으로 구현
  * 사용자와의 이전 대화를 기록하는 [메모리 (Memory)](../LangChain_메모리.md) 기능 포함

## 2. 개발 일정

* 전체 개발 일정
  * **2026.02.19 (목) - 02.28 (토), 9 days**
* 상세 개발 일정

| 구분         | 계획 내용                              | 일정                     | branch                                   | issue                                                          | 상태 |
|------------|------------------------------------|------------------------|------------------------------------------|----------------------------------------------------------------|----|
| 📃 문서화     | 기본 요구사항 및 개발 일정 문서화                | 02.19 목 (1d)           |                                          |                                                                | ✅  |
| 🧠 모델 선택   | 적절한 한국어 LLM 순위 산출 (최신 Open-Source) | 02.19 목 (1d)           |                                          |                                                                | ✅  |
| 🔨 모델 구현   | 도구 함수 구현 (전/후 날짜 계산, 요일 계산)        | 02.20 금 (1d)           | ```LangChain-practice-001-tool```        | [issue](https://github.com/WannaBeSuperteur/AI-study/issues/1) | ✅  |
| 🔨 모델 구현   | 메모리 구현 (도구 함수와 동일 방식)              | 02.20 금 (1d)           | ```LangChain-practice-002-memory```      | [issue](https://github.com/WannaBeSuperteur/AI-study/issues/2) | ✅  |
| 🧠 모델 선택   | LLM 학습 (Fine-Tuning) 대상 LLM 최종 선택  | 02.20 금 (1d)           | ```LangChain-practice-003-fine-tuning``` | [issue](https://github.com/WannaBeSuperteur/AI-study/issues/3) | ✅  |
| 📝 데이터셋 제작 | LLM 학습 데이터셋 제작                     | 02.20 금 (1d)           | ```LangChain-practice-003-fine-tuning``` | [issue](https://github.com/WannaBeSuperteur/AI-study/issues/3) | ✅  |
| 🧪 모델 학습   | LLM 학습 (Fine-Tuning) 실시            | 02.20 금 - 02.22 일 (3d) | ```LangChain-practice-003-fine-tuning``` | [issue](https://github.com/WannaBeSuperteur/AI-study/issues/3) | ✅  |
| ⚙ 기능 구현    | LLM 에이전트 기능 구현                     | 02.22 일 - 02.26 목 (5d) | ```LangChain-practice-004-agent```       | [issue](https://github.com/WannaBeSuperteur/AI-study/issues/4) | ✅  |
| ⚙ 기능 구현    | LLM 에이전트 기능 구현 (tool call 재 구현)    | 02.23 월 - 02.26 목 (5d) | ```LangChain-practice-005-tool-call```   | [issue](https://github.com/WannaBeSuperteur/AI-study/issues/4) | ✅  |
| 🔍 최종 검토   | 최종 QA (버그 유무 검사)                   | 02.28 토 (1d)           |                                          |                                                                | ⬜  |
| 📃 문서화     | 프로젝트 문서 정리 및 마무리                   | 02.28 토 (1d)           |                                          |                                                                | ⬜  |

## 3. 구현 내용 상세

* 전체 구조

TBU

* 학습 데이터셋
  * [toolcall_training_data.csv](toolcall_training_data.csv)

| 컬럼                     | 설명                                  | LLM 학습 데이터로 사용                                                       |
|------------------------|-------------------------------------|----------------------------------------------------------------------|
| ```user_input```       | 사용자의 최초 입력 (날짜 또는 요일 계산 요청)         | - **Tool Call 실시** LLM의 **입력** 데이터<br>- **최종 답변 생성** LLM의 **입력** 데이터 |
| ```tool_call_output``` | LangChain의 tool call을 위한 LLM output | **Tool Call 실시** LLM의 **출력** 데이터                                     |
| ```tool_call_result``` | tool call 결과로 반환되는 값 (날짜 또는 요일)     | **최종 답변 생성** LLM의 **입력** 데이터                                         |
| ```final_output```     | tool call 결과를 해석한 LLM의 최종 답변        | **최종 답변 생성** LLM의 **출력** 데이터                                         |

### 3-1. 학습 데이터셋 생성 과정

* 요약
  * ChatGPT 를 이용하여 200 rows 규모의 학습 데이터셋을 빠르게 생성
* 사용 모델
  * ChatGPT 5.2 Thinking
* 사용 프롬프트
  * 최초 프롬프트로 csv 파일 생성 이후에도, 생성된 csv 파일의 오류를 지속적으로 개선 요청

```
위 내용을 참고하여, 예를 들어 다음과 같은 식으로 200개의 행이 있는 학습 데이터를 csv 형식으로 만들어줘.
 - 사용자 질문 예시 (최대한 다양하게) (열 이름: user_input)
   - 2026년 2월 20일이 무슨 요일이야?
   - 2026년 1월 15일의 요일을 알려줘
   - 2026년 12월 25일부터 10일 후는 언제야?
   - 2025년 1월 1일부터 100일 후 날짜를 언제인지 알려줘
 - Tool Call을 할 수 있는 LLM 응답 (열 이름: tool_call_output)
 - Tool Call 결과 (열 이름: tool_call_result)
 - Tool Call 함수를 받아서 최종 응답 (열 이름: final_output)
   - 2026년 2월 20일은 금요일입니다. 혹시 추가 질문 있으신가요?
   - 그날은 목요일이에요! 혹시 더 궁금한 거 있어요?
   - 2027년 1월 4일! 맞죠?
   - 2025년 4월 11일인데 그날 무슨 이벤트 있어요? 궁금해요!
```

```
여기서 tool_call_result 열의 값이 날짜 계산 결과인 경우 예를 들어 '2026년 2월 20일'이 아닌 '2026년 02월 20일'처럼 표시되는데, 이를 '2026년 2월 20일' 형식으로 수정해 줘.
```

```
다음과 같은 오류가 계속 있는데, 이를 수정해줘.
 - 요일 계산 결과 중 'O요일네요' -> 'O요일이네요' 로 수정
 - final_output 컬럼에서 '참고로 날짜 형식은'과 같은 날짜 형식에 대한 언급은 생략한다.
```

```
다음과 같은 오류가 계속 있는데, 이를 수정해줘.
 - N일 전의 경우 -N을 인수로 넣고 N일 후의 경우 N을 인수로 넣어야 하는데, 반대로 되어 있음
```

## 4. LLM 선택

* 한국어 LLM 모델 선택
  * **최종 채택: [Midm-2.0-Mini-Instruct](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct)**
  * [Dnotitia LLM 한국어 리더보드 (한국 모델)](https://leaderboard.dnotitia.com/?filter=korea) 참고

### 4-1. 한국어 LLM 선택 이유

**1. 한국어 LLM 선택 절차**

* **1.** [Dnotitia LLM 한국어 리더보드 (한국 모델)](https://leaderboard.dnotitia.com/?filter=korea) 에서 **LLM 브랜드명 (예: ```kakakcorp/kanana```)** 추출
  * 추출 결과 (총 8개 브랜드)
    * ```naver/HyperCLOVAX``` + ```naver/HCX```
    * ```kakaocorp/kanana```
    * ```LGAI-EXAONE```
    * ```skt/A.X```
    * ```KT/Midm-2.0```
    * ```dnotitia/DNA-2.0```
    * ```upstage/solar```
    * ```trillionlabs/Tri```

* **2.** 추출한 LLM 브랜드명을 HuggingFace 로 검색해서, 해당 브랜드의 **모든 모델** 탐색
  * 이때 **5B 이하** 의 모델 탐색

| LLM 브랜드                                   | HuggingFace 탐색 결과 (5B 이하 모델)                                                                                                                                           | [브랜드 순위 (참고)](#3-2-한국어-llm-성능-상세-비교-결과) |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| ```naver/HyperCLOVAX``` + ```naver/HCX``` | [HyperCLOVAX-SEED-Text-Instruct-1.5B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)                                                    |                                         |
| ```kakaocorp/kanana```                    | [kanana-1.5-2.1b-instruct-2505](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505)                                                                        | 2위                                      |
| ```LGAI-EXAONE```                         | - [EXAONE-Deep-2.4B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B)<br>- [EXAONE-3.5-2.4B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct) |                                         |
| ```skt/A.X```                             | [ko-gpt-trinity-1.2B-v0.5](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5)                                                                                        | 3위                                      |
| ```KT/Midm-2.0```                         | [Midm-2.0-Mini-Instruct](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct)                                                                                 | **1위** (✅ 최종 채택)                        |
| ```dnotitia/DNA-2.0```                    | [DNA-2.0-4B](https://huggingface.co/dnotitia/DNA-2.0-4B)                                                                                                               |                                         |
| ```upstage/solar```                       | (5B 미만 LLM 없음)                                                                                                                                                         |                                         |
| ```trillionlabs/Tri```                    | [Tri-1.9B-Base](https://huggingface.co/trillionlabs/Tri-1.9B-Base)                                                                                                     |                                         |

* **3.** 위 한국어 모델 리더보드를 기준으로 **브랜드 별 LLM 성능 추이** 를 비교
  * 해당 비교 결과에 근거하여 최종적으로 **5.0B 미만 파라미터 개수 구간** 에서 가장 성능이 좋을 법한 브랜드의 모델 선정
  * 해당 선택한 모델이 **OOM, 권한 오류 등 오류** 발생 시, 그 다음으로 성능이 좋은 모델을 **오류 없는 모델이 처음으로 나타날 때까지** 선택
  * 아래의 [4-2. 한국어 LLM 성능 상세 비교 결과](#4-2-한국어-llm-성능-상세-비교-결과) 참고

**2. 해당 절차로 진행한 이유**

* Local PC (12GB GPU) 에서 LLM을 Fine-Tuning 할 때, **5B 초과의 LLM의 경우 OOM (Out of Memory) 가능성**
  * 메모리 및 연산량 절약을 위해, Fine-Tuning 시 **[LoRA](../../AI%20Basics/LLM%20Basics/LLM_기초_Fine_Tuning_LoRA_QLoRA.md) + [Quantization](../../AI%20Basics/LLM%20Basics/LLM_기초_Quantization.md)** 적용 시에도 OOM 가능 예상

* 리더보드 선택 이유
  * 리더보드 설명에 따르면 아래와 같이 **RAG, 툴 콜링 등의 성능을 정밀하게 평가** 하므로, 
  * Tool Call 이 가능한 LLM Agent 성능 평가지표로서 적합

> 디노티시아는 한국어 기반 추론, RAG, 툴 콜링 등의 성능을 정밀하게 평가하기 위해 자체 벤치마크를 구축했습니다.

### 4-2. 한국어 LLM 성능 상세 비교 결과

* 최종 비교 결과 (브랜드 별 순위)
  * **KT Midm (1위)** > Kakao Kanana (2위) > SKT A.X (3위)
  * 4위 이후
    * Dnotitia > TrillionLabs > Naver > Upstage > LG EXAONE 

![image](../images/LangChain_Practice_1.PNG)

* 전체 리스트
  * 출처: [Dnotitia LLM 한국어 리더보드 (한국 모델)](https://leaderboard.dnotitia.com/?filter=korea) (2026.02.19)
  * **총 27개** LLM 확인

| LLM                                           | 브랜드/회사             | LLM 크기 (파라미터 개수)     | LeaderBoard 점수 |
|-----------------------------------------------|--------------------|----------------------|----------------|
| ```naver/HyperCLOVAX-SEED-Think-14B```        | ```naver```        | 14.0B                | 0.729          |
| ```naver/HCX-DASH-002```                      | ```naver```        | (확인 불가)              | 0.570          |
| ```naver/HCX-003```                           | ```naver```        | (확인 불가)              | 0.614          |
| ```naver/HCX-005```                           | ```naver```        | (확인 불가)              | 0.700          |
| ```naver/HCX-007```                           | ```naver```        | (확인 불가)              | 0.852          |
| ```kakaocorp/kanana-1.5-8b-instruct-2505```   | ```kakaocorp```    | 8.0B                 | 0.703          |
| ```kakaocorp/kanana-1.5-15.7b-a3b-instruct``` | ```kakaocorp```    | 15.7B                | 0.719          |
| ```LGAI-EXAONE/EXAONE-3.5-32B```              | ```LGAI-EXAONE```  | 32.0B                | 0.784          |
| ```LGAI-EXAONE/EXAONE-Deep-32B```             | ```LGAI-EXAONE```  | 32.0B                | 0.707          |
| ```LGAI-EXAONE/EXAONE-4.0-32B```              | ```LGAI-EXAONE```  | 32.0B                | 0.719          |
| ```LGAI-EXAONE/K-EXAONE-236B-A23B```          | ```LGAI-EXAONE```  | 23.0B (total 236.0B) | 0.831          |
| ```skt/A.X-3.1-Light```                       | ```skt/A.X```      | 7.0B                 | 0.692          |
| ```skt/A.X-3.1```                             | ```skt/A.X```      | 35.0B                | 0.781          |
| ```skt/A.X-4.0-Light```                       | ```skt/A.X```      | 7.0B                 | 0.680          |
| ```skt/A.X-4.0```                             | ```skt/A.X```      | 72.0B                | 0.874          |
| ```skt/A.X-K1```                              | ```skt/A.X```      | 519.0B               | 0.881          |
| ```KT/Midm-2.0-Mini-Instruct```               | ```KT/Midm-2.0```  | 2.3B                 | 0.551          |
| ```KT/Midm-2.0-Base-Instruct```               | ```KT/Midm-2.0```  | 8.0B                 | 0.739          |
| ```dnotitia/DNA-2.0-14B```                    | ```dnotitia```     | 15.0B                | 0.768          |
| ```dnotitia/DNA-2.0-30B-A3B```                | ```dnotitia```     | 3.0B (total 31.0B)   | 0.808          |
| ```dnotitia/DNA-2.0-235B-A22B```              | ```dnotitia```     | 22.0B (total 235.0B) | 0.865          |
| ```upstage/solar-pro```                       | ```upstage```      | 22.0B                | 0.660          |
| ```upstage/solar-pro2```                      | ```upstage```      | 30.9B                | 0.852          |
| ```upstage/Solar-Open-100B```                 | ```upstage```      | 102.0B               | 0.811          |
| ```upstage/solar-pro3```                      | ```upstage```      | 12.0B (total 102.0B) | 0.842          |
| ```trillionlabs/Tri-7B```                     | ```trillionlabs``` | 8.0B                 | 0.685          |
| ```trillionlabs/Tri-21B```                    | ```trillionlabs``` | 21.0B                | 0.793          |

## 5. 이슈 사항 및 해결 방법

### 5-1. EOS token 학습 안됨

* 문제 상황
  * LLM 학습 시 EOS token이 학습이 되지 않아서, **inference 시 EOS token 이 생성되지 않아서 token limit 까지 계속 생성**
* 문제 원인
  * LLM tokenizer 의 ```pad_token``` 이 ```eos_token``` 과 같은 경우,
  * ```DataCollatorForCompletionOnlyLM``` 에 의해 **해당 토큰이 ```-100``` 으로 라벨링되어 학습 불가**
* 해결 방법
  * tokenizer 의 ```pad_token``` 과 ```eos_token``` 이 서로 동일할 경우, **서로 다르게 설정** 

```python
if tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.pad_token = '<pad>'
```

### 5-2. LLM Fine-Tuning 후, 응답이 제대로 생성되지 않음

* 문제 상황
  * [5-1. EOS token 학습 안됨](#5-1-eos-token-학습-안됨) 해결 이후
  * LLM Fine-Tuning 후, 해당 LLM으로부터 **응답이 제대로 생성되지 않음**
    * 응답 시작 시점에 ```eos_token``` 이 자주 등장하여, LLM에 의해 추가 생성되는 내용이 사실상 없는 경우가 많음
* 문제 원인 **(추정)**
  * LLM의 학습 데이터는 ```(입력 데이터) ### Answer: (출력 데이터)``` 꼴 
  * LLM이 그 중간의 ```response_template``` (= ``` ### Answer:```) 에 해당하는 부분을 **학습 시 무시** 하는 것으로 추정
    * 근거: 기존 문장에 이어지는 듯한 내용이 생성되곤 함 (예: ```안녕 반가워``` → ```!```가 추가 생성) 
* 해결 방법
  * 기존 Oh-LoRA 프로젝트를 참고하여, **답변 시작 지점 = 사용자 프롬프트 끝 지점** 에 ```(답변 시작)``` 추가

```python
dataset_df['text'] = dataset_df.apply(
    lambda x: f"{x['input_data']}{ANSWER_PREFIX} {ANSWER_START_MARK} {x['output_data']}{tokenizer.eos_token}",
    axis=1
)
dataset = generate_llm_trainable_dataset(dataset_df)
```

```python
def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    global lora_llm, tokenizer

    print('=== INFERENCE TEST ===')

    for valid_input in self.valid_dataset:
        valid_input_text = valid_input['text'].split(ANSWER_PREFIX)[0] + f' {ANSWER_PREFIX}'
```

### 5-3. LLM output 에서 처음에 EOS token 발생

* 문제 상황
  * LLM 이 생성하는 output 에서 처음에 EOS token 발생하여, **사실상 아무것도 생성되지 않음**
* 해결 방법
  * LLM 을 이용한 생성 시, 아래와 같이 **최소 생성 token 개수** 를 지정

```python
outputs = lora_llm.generate(**inputs,
                            max_length=self.max_length,
                            do_sample=True,
                            temperature=0.6,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            min_new_tokens=5)                      # 처음에 바로 EOS token 이 생성되는 것 방지
```

### 5-4. Fine-Tuning 된 LLM 로딩 시 tensor size 불일치

* 문제 상황
  * Fine-Tuning 된 LLM 로딩 시, tensor 크기가 불일치하여 다음과 같은 오류 발생

```
RuntimeError: Error(s) in loading state_dict for LlamaForCausalLM:
        size mismatch for model.embed_tokens.weight: copying a param with shape torch.Size([131384, 1792]) from checkpoint, the shape in current model is torch.Size([131392, 1792]).
        size mismatch for lm_head.weight: copying a param with shape torch.Size([131384, 1792]) from checkpoint, the shape in current model is torch.Size([131392, 1792]).
```

* 문제 원인
  * Fine-Tuning 된 LLM과 원본 Mi:dm-2.0 LLM 간 **tokenizer 의 vocab size 불일치**
* 해결 방법
  * **config 에서 ```vocab_size```를 수정** (Fine-Tuning 된 LLM 의 vocab size)
  * ```ignore_mismatched_sizes=True``` 로 **크기 불일치 시에도 오류 미 발생** 하도록 수정 + 후처리

```python
config = AutoConfig.from_pretrained(ORIGINAL_MIDM_LLM_PATH)
config.vocab_size = len(tokenizer)  # new vocab size

llm = AutoModelForCausalLM.from_pretrained(
    llm_path,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    ignore_mismatched_sizes=True
)
```

## 6. 참고

### 6-1. Quantization 적용/미 적용 시 GPU 메모리 사용량 차이

* 적용 Quantization
  * **BitsAndBytesConfig**

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='bfloat16'
)
```

* Quantization 적용 vs. 미 적용 시 GPU 메모리 사용량
  * 2개의 [Midm-2.0-Mini-Instruct](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct) Fine-Tuning 된 LLM 로딩 시 기준

| Quantization 적용 시 | Quantization 미 적용 시 | 차이        |
|-------------------|---------------------|-----------|
| 9,970 MB          | 6,141 MB            | 🔻 38.4 % |
