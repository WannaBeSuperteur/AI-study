## 목차

* 목차
  * [1. Transformers LLM 연동 개요](#1-transformers-llm-연동-개요)
  * [2. Transformers LLM 연동 과정](#2-transformers-llm-연동-과정)
  * [3. Transformers 연동 LLM 동작 확인](#3-transformers-연동-llm-동작-확인)
* [ipynb 실습 파일](ipynb/LangChain_LLM_Transformers_연동.ipynb)

## 1. Transformers LLM 연동 개요

* Transformers 라이브러리를 이용하여 학습한 LLM을 LangChain과 연동한다.
* Transformers 라이브러리를 통해 **자체적으로 Fine-Tuning 한 모델** 을 연동하므로, 다음과 같은 장점을 얻을 수 있다.
  * OpenAI, Claude 등 **상용 LLM 서비스의 token 당 사용료 지불 불필요**
  * **target task 에 더욱 적합** 한, Fine-Tuning 된 LLM으로 사용 가능

## 2. Transformers LLM 연동 과정

* Transformers 로 학습된 모델의 **Fine-Tuning 이전 원본 모델** 다운로드 (필요 시)

```
# Download Kanana-1.5 2.1B Base Original Model

!huggingface-cli download kakaocorp/kanana-1.5-2.1b-base --local-dir kakao_original
```

* Transformers 로 학습된 모델 로딩

```python
# load Oh-LoRA original LLM

import os
print(os.path.abspath('ohlora_llm'))

llm_path = 'ohlora_llm'
ohlora_llm = AutoModelForCausalLM.from_pretrained(
    llm_path,
    trust_remote_code=True,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(llm_path)

pipe = pipeline(
    "text-generation",
    model=ohlora_llm,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.6,
    top_p=0.95
)
```

* LangChain LLM 으로 convert 실시

```python
# convert to LangChain LLM

local_llm = HuggingFacePipeline(pipeline=pipe)
```

## 3. Transformers 연동 LLM 동작 확인

* 본 실습 예제에서는 **Oh-LoRA v3 (2025.05.26 - 06.05)** 에서 학습한 LLM 사용
  * [Oh-LoRA v3 프로젝트 바로가기](https://github.com/WannaBeSuperteur/AI_Projects/tree/main/2025_05_26_OhLoRA_v3)

```python
# test

user_message = '아 오늘 뭐 입고 나가지'
final_llm_prompt = f'[오늘 일정: 친구랑 카페 방문] (지금은 화요일 오전) {user_message} (답변 시작)'
llm_answer = local_llm.invoke(final_llm_prompt)
```

