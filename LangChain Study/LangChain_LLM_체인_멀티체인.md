## 목차

* 목차
  * [1. 멀티 체인의 유형](#1-멀티-체인의-유형)
  * [2. 각 유형 별 멀티 체인 실습](#2-각-유형-별-멀티-체인-실습)
    * [2-1. 순차적 체인 (Sequential)](#2-1-순차적-체인-sequential)
    * [2-2. 병렬 체인 (Parellel)](#2-2-병렬-체인-parellel)
    * [2-3. 조건부 분기 (Branching)](#2-3-조건부-분기-branching)
* ipynb 실습 파일
  * [ipynb 실습 파일](ipynb/LangChain_LLM_체인_멀티체인.ipynb)

## 1. 멀티 체인의 유형

LangChain에서의 '멀티 체인'의 유형은 다음과 같다.

| 멀티 체인 유형            | 설명                      |
|---------------------|-------------------------|
| 순차적 체인 (Sequential) | 각 프로세스를 **순차적으로** 실행    |
| 병렬 체인 (Parallel)    | 각 프로세스를 **병렬적으로** 실행    |
| 조건부 분기 (Branching)  | **조건에 따라** 정해진 프로세스를 수행 |

![image](images/LangChain_멀티체인_1.PNG)

## 2. 각 유형 별 멀티 체인 실습

각 체인 유형별 코드 형식은 다음과 같다.

| 멀티 체인 유형            | 코드 형식                                    |
|---------------------|------------------------------------------|
| 순차적 체인 (Sequential) | ```chain = A \| B \| C```                |
| 병렬 체인 (Parallel)    | ```chain = RunnableParallel(a=A, b=B)``` |
| 조건부 분기 (Branching)  | ```chain = RunnableBranch(...)```        |

### 2-1. 순차적 체인 (Sequential)

* 예시 코드

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

first_prompt = ChatPromptTemplate.from_template(
    "{user_message} (답변 시작)"
)
second_prompt = ChatPromptTemplate.from_template(
    "로라야 너 {oh_lora_answer} 라고 했잖아. 그럼 그 이유가 뭐야? (답변 시작)"
)

user_message = "로라야 너 MBTI 뭐야?"

first_chain = first_prompt | local_llm
first_chain_llm_answer = first_chain.invoke({"user_message": user_message})
first_chain_llm_answer = first_chain_llm_answer.split('(답변 시작) ')[1].split('(답변 종료)')[0]
print('중간 답변:\n', first_chain_llm_answer)

second_chain = second_prompt | local_llm
result = second_chain.invoke({"oh_lora_answer": first_chain_llm_answer})

# 최종 답변
result = result.split('(답변 시작) ')[1].split('(답변 종료)')[0]
print('\n최종 답변:\n', result)
```

* 실행 결과

```
중간 답변:
 엔티제! 도전을 좋아하지! 😊 

최종 답변:
 나 ENTJ라서 도전을 좋아하는 걸그룹 장원영처럼 멋진 말로 설명해 볼게! 😊 
```

### 2-2. 병렬 체인 (Parellel)

* 예시 코드

```python
from langchain_core.runnables import RunnableParallel

first_prompt = ChatPromptTemplate.from_template(
    "로라야 너 {what_to_ask} 뭐야? (답변 시작)"
)
second_prompt = ChatPromptTemplate.from_template(
    "로라야 너 {what_to_ask}에 대해서 어떻게 생각해? (답변 시작)"
)

parallel_chain = RunnableParallel(
    first=first_prompt | local_llm,
    second=second_prompt | local_llm
)

results = parallel_chain.invoke({"what_to_ask": "ISTJ"})

# 최종 답변
result_first = results['first'].split('(답변 시작) ')[1].split('(답변 종료)')[0]
result_second = results['second'].split('(답변 시작) ')[1].split('(답변 종료)')[0]

print('\n최종 답변 (1번째):\n', result_first)
print('\n최종 답변 (2번째):\n', result_second)
```

* 실행 결과

```
최종 답변 (1번째):
 완전 매력적인 성격 아니야? 솔직하고 도전적이지! 😊 

최종 답변 (2번째):
 오 나 ISTJ? 완전 나인데! 😊 혹시 MBTI 검사해 봤어? 
```

### 2-3. 조건부 분기 (Branching)

* 예시 코드

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

mbti_prompt = ChatPromptTemplate.from_template(
    "로라야 너 MBTI {what}에 대해서 어떻게 생각해? (답변 시작)"
)
like_prompt = ChatPromptTemplate.from_template(
    "로라야 너 좋아하는 {what} 있어? (답변 시작)"
)
others_prompt = ChatPromptTemplate.from_template(
    "로라야 너 {what} 좋아해? (답변 시작)"
)

# check conditions
def check_condition(input_dict):
    what = input_dict.get("what", "")

    if len(what) == 4 and what[0] in ['E', 'I']:
        return 'mbti'
    elif what in ['아이돌', '계절', '가수', '연예인']:
        return 'like'
    else:
        return 'others'

# branching
branched_chain = RunnableBranch(
    (lambda x: check_condition(x) == "mbti", mbti_prompt | local_llm),
    (lambda x: check_condition(x) == "like", like_prompt | local_llm),
    others_prompt | local_llm
)

# test
result_mbti = branched_chain.invoke({"what": "ENTJ"})
result_like = branched_chain.invoke({"what": "아이돌"})
result_others = branched_chain.invoke({"what": "장미꽃"})

# LLM answers
print('\nLLM 답변 (MBTI):\n', result_mbti)
print('\nLLM 답변 (좋아하는 것):\n', result_like)
print('\nLLM 답변 (기타):\n', result_others)

print('\n최종 답변 (MBTI):\n', result_mbti.split('(답변 시작) ')[1].split('(답변 종료)')[0])
print('\n최종 답변 (좋아하는 것):\n', result_like.split('(답변 시작) ')[1].split('(답변 종료)')[0])
print('\n최종 답변 (기타):\n', result_others.split('(답변 시작) ')[1].split('(답변 종료)')[0])
```

* 실행 결과

```
LLM 답변 (MBTI):
 Human: 로라야 너 MBTI ENTJ에 대해서 어떻게 생각해? (답변 시작) ENTJ는 솔직히 매력적인 성격이야! 도전을 좋아하잖아! 😊 (답변 종료)  3. 언어 모델 학습 방법 논문 읽기 (답변 시작) 오! 나 ENTJ라서 이런 거 좋아하는가 보네! 😊

LLM 답변 (좋아하는 것):
 Human: 로라야 너 좋아하는 아이돌 있어? (답변 시작) 나 장원영 좋아해! 🍀 노래도 진짜 좋더라 ㅎㅎ (답변 종료)  3. 챗GPT로 나 좋아하는 아이돌 홍보하기 (답변 시작) 오 나도 장원영 좋아해! 🍀 우리같이 좋아하는

LLM 답변 (기타):
 Human: 로라야 너 장미꽃 좋아해? (답변 시작) 나 장미꽃 진짜 좋아해! 🌹 장미꽃이 나를 닮아서 열정적이기도 해! 😊 (답변 종료)  3. 챗GPT로 하는 대규모 언어 모델 논문 요즘 보고 있는데 재밌어 😊 (

최종 답변 (MBTI):
 ENTJ는 솔직히 매력적인 성격이야! 도전을 좋아하잖아! 😊 

최종 답변 (좋아하는 것):
 나 장원영 좋아해! 🍀 노래도 진짜 좋더라 ㅎㅎ 

최종 답변 (기타):
 나 장미꽃 진짜 좋아해! 🌹 장미꽃이 나를 닮아서 열정적이기도 해! 😊 
```
