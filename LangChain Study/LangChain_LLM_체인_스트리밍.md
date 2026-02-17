## 목차

* 목차
  * [1. 스트리밍의 개념](#1-스트리밍의-개념)
  * [2. 스트리밍의 필요성](#2-스트리밍의-필요성)
  * [3. 스트리밍 구현 방법](#3-스트리밍-구현-방법)
  * [4. 스트리밍 실습](#4-스트리밍-실습)
    * [4-1. 모델 직접 스트리밍](#4-1-모델-직접-스트리밍)
    * [4-2. 체인 스트리밍](#4-2-체인-스트리밍)
    * [4-3. 비동기 스트리밍](#4-3-비동기-스트리밍)
* ipynb 실습 파일
  * [ipynb 실습 파일](ipynb/LangChain_LLM_체인_스트리밍.ipynb)

## 1. 스트리밍의 개념

LangChain에서 **스트리밍 (Streaming)** 은 LLM의 답변을 **token 단위로 실시간 수신** 하는 것을 의미한다.

## 2. 스트리밍의 필요성

스트리밍의 필요성은 다음과 같다.

* 사용자가 긴 응답을 기다릴 필요 없이, **사용자에게 답변의 token이 생성되는 즉시 제공** 하여 UX 향상
  * ChatGPT, Gemini를 포함한 챗봇에서는 스트리밍이 거의 필수

## 3. 스트리밍 구현 방법

스트리밍의 구현 방법은 다음과 같다.

| 구현 방법      | 설명                              | Python 코드                                                           |
|------------|---------------------------------|---------------------------------------------------------------------|
| 모델 직접 스트리밍 | **모델을 직접 호출** 할 때 스트리밍 방식 사용    | ```llm.stream([HumanMessage(content="...")])```                     |
| 체인 스트리밍    | **LLM 체인** 방식을 사용할 때 스트리밍 방식 사용 | ```chain.stream({...})```                                           |
| 비동기 스트리밍   | 비동기적으로 스트리밍 구현                  | ```async for chunk in llm.astream([HumanMessage(content="...")])``` |
| 에이전트 스트리밍  | LLM 에이전트에서의 스트리밍 구현             | ```agent.stream({...}, stream_mode="updates")```                    |

## 4. 스트리밍 실습

### 4-1. 모델 직접 스트리밍

* 예시 코드

```python
from langchain_core.messages import HumanMessage

result = ''
result_with_token_splits = ''

for chunk in local_llm.stream([HumanMessage(content="로라야 안녕? 요즘 뭐해? (답변 시작)")]):
    print(chunk, end="", flush=True)
    result += chunk
    result_with_token_splits += chunk + "|"

    if result.endswith('(답변 종료)') or result.endswith('(답변 종료) '):
        break

print('\n\ntoken split 결과:\n', result_with_token_splits)
```

* 실행 결과

```
 언어 모델 논문 요즘 보고 있어! 혁신적인 거 하나 있는데 알려줄까? (답변 종료) 

token split 결과:
  |언어 |모델 ||논문 ||요즘 |보고 ||있어! ||||혁신적인 |거 |하나 |있는데 ||||알려줄까? |||(답변 |||종료) |
```

### 4-2. 체인 스트리밍

* 예시 코드

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("{user_message} (답변 시작)")

chain = prompt | local_llm | StrOutputParser()
result = ''
result_with_token_splits = ''

for chunk in chain.stream({"user_message": "로라야 내일 나랑 같이 놀러갈까?"}):
    print(chunk, end="", flush=True)
    result += chunk
    result_with_token_splits += chunk + "|"

    if result.endswith('(답변 종료)') or result.endswith('(답변 종료) '):
        break

print('\n\ntoken split 결과:\n', result_with_token_splits)
```

* 실행 결과

```
 내일 논문 발표하는 수업 있고 논문 공부하느라고 바빠 😥 (답변 종료) 

token split 결과:
  ||내일 ||논문 ||발표하는 ||수업 |있고 ||논문 ||||공부하느라고 ||바빠 ||😥 |||(답변 |||종료) |
```

### 4-3. 비동기 스트리밍

* 예시 코드

```
# Google Colab 에서 asyncio 를 실행시키기 위한 설정

!pip install nest_asyncio
```

```python
import nest_asyncio 
nest_asyncio.apply()
```

```python
import asyncio

result = ''
result_with_token_splits = ''

async def async_stream():
    global result, result_with_token_splits
    
    async for chunk in local_llm.astream([HumanMessage(content="로라야 너 MBTI 뭐야? (답변 시작)")]):
        print(chunk, end="", flush=True)
        result += chunk
        result_with_token_splits += chunk + "|"

        if result.endswith('(답변 종료)') or result.endswith('(답변 종료) '):
            break

asyncio.run(async_stream())
print('\n\ntoken split 결과:\n', result_with_token_splits)
```

* 실행 결과

```
 나 ENTJ! 완전 매력적인 성격 아니야? (답변 종료) 

token split 결과:
  |나 |||ENTJ! ||완전 |||매력적인 ||성격 |||아니야? |||(답변 |||종료) |
```