## 목차

* 목차
  * [1. LangChain의 메모리 개요 및 필요성](#1-langchain의-메모리-개요-및-필요성)
  * [2. LangChain의 메모리 유형](#2-langchain의-메모리-유형)
  * [3. 메모리의 구현 방식](#3-메모리의-구현-방식)
    * [3-1. RunnableWithMessageHistory](#3-1-runnablewithmessagehistory)
    * [3-2. LangGraph 기반 방법](#3-2-langgraph-기반-방법)
  * [4. 실전 예제](#4-실전-예제)
    * [4-1. 단기 메모리](#4-1-단기-메모리)
    * [4-2. 장기 메모리](#4-2-장기-메모리)
* ipynb 실습 파일
  * [LangGraph 실습 파일](ipynb/LangChain_메모리_LangGraph_기반.ipynb)
  * [장기 메모리 예제](ipynb/LangChain_메모리_장기_메모리_예제.ipynb)

## 1. LangChain의 메모리 개요 및 필요성

LLM에서 쓰이는 **메모리 (Memory) 시스템** 은 LLM의 대화 기록을 저장하기 위한 시스템이다.

* LLM은 기본적으로 **이전 상태를 저장하지 않아서, 각 프롬프트가 이전 대화 내역과 상관없이 독립적으로 처리** 된다.
* 이로 인해, **이전 대화 내용의 맥락을 반영하지 않는 답변** 이 생성된다.

메모리 시스템의 역할은 **대화 기록 저장, 문맥 유지 + 개인화 (사용자 개별 세션 관리)** 이다.

## 2. LangChain의 메모리 유형

LangChain의 메모리 패턴 유형은 다음과 같다.

| 메모리 패턴 유형                  | 설명                            | 사용 사례                                  |
|----------------------------|-------------------------------|----------------------------------------|
| 단기 메모리 (Short-term Memory) | **현재 대화 세션** 동안 유지되는 메모리      | 이전 대화 요약 등                             |
| 장기 메모리 (Long-term Memory)  | **영구적으로 (여러 세션 동안)** 유지되는 메모리 | 필요한 정보 추출 (Vector DB), 사용자 프로필 정보 연계 등 |

## 3. 메모리의 구현 방식

LangChain 메모리의 구현 방식은 다음과 같다.

| 메모리 구현 방식                        | 설명                                                |
|----------------------------------|---------------------------------------------------|
| ```RunnableWithMessageHistory``` | 간단한 챗봇 애플리케이션에 적합                                 |
| LangGraph 기반                     | 복잡한 에이전트 (상태 관리 등) 에 적합<br>- **LangChain 1.0 권장** |

### 3-1. RunnableWithMessageHistory

**RunnableWithMessageHistory** 구현 방식의 특징은 다음과 같다.

* 매 답변 생성 시작 시마다 **전체 대화 컨텍스트가 LLM에 전달** 됨
* **사용자 메시지 + AI 응답** 의 모든 이전 대화가 컨텍스트에 저장됨
* 복잡한 상태 관리가 필요없는, 간단한 챗봇 앱에 적합

구현상의 특징은 다음과 같다.

* ```InMemoryChatMessageHistory``` 를 이용하여 메모리에 대화 기록 저장
* **Session ID** 를 이용하여 사용자/세션 구분
  * 각 세션마다 대화 기록을 독립적으로 유지
* **In-memory** 방식으로, **프로세스 종료 시 대화 기록 소멸**

### 3-2. LangGraph 기반 방법

**LangGraph 기반 방법** 구현 방식의 특징은 다음과 같다.

* **복잡한 상태 관리 기능** 이 필요한 LLM Agent 에 적합
* LangChain 1.0 에서 권장하는 방법

**예시 코드**

* 필요한 라이브러리 import

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
```

* Memory Saver 생성

```python
# create memory saver
checkpointer = MemorySaver()

# create agent
agent = create_agent(
    local_llm,
    tools=[],
    checkpointer=checkpointer
)

# config
config = {"configurable": {"thread_id": "conversation-1"}}
```

* 실제 대화 진행

```python
# 1번째 대화

result1 = agent.invoke(
    {"messages": [
        {"role": "user",
         "content": "로라야 나 장원영 좋아해. 너도? (답변 시작)"}
    ]},
    config=config
)
result1_answer = result1["messages"][-1].content
result1_answer_final = result1_answer.split('(답변 시작)')[2].split('(답변 종료)')[0]

result1_answer_final
```

결과: ```나도 장원영 좋아해! 완전 럭키비키라는 말 알아? 🍀```

```python
# 2번째 대화

result2 = agent.invoke(
    {"messages": [
        {"role": "user",
         "content": "로라야 내가 누구 좋아한다고 했지? (답변 시작)"}
    ]},
    config=config
)
result2_answer = result2["messages"][-1].content
result2_answer_final = result2_answer.split('(답변 시작)')[2].split('(답변 종료)')[0]

result2_answer_final
```

결과: ```나도 장원영 좋아해! 완전 럭키비키라는 말 알아? 🍀```

## 4. 실전 예제

### 4-1. 단기 메모리

**단기 메모리 (Short-term Memory)** 는 **단일 대화 세션** 에서의 사용자-LLM 간 이전 상호작용을 기억한다.

* LangGraph 기반의 MemorySaver 방법 사용
* LLM Agent 를 호출하는 함수 ```create_agent``` 에 ```checkpointer``` 를 전달하면 됨
* 참고
  * [3-2. LangGraph 기반 방법](#3-2-langgraph-기반-방법)

```python
# create memory saver
checkpointer = MemorySaver()

# create agent
agent = create_agent(
    local_llm,
    tools=[],
    checkpointer=checkpointer
)
```

### 4-2. 장기 메모리

**장기 메모리 (Long-term Memory)** 는 대화 세션이 종료되어도 **영구적으로 정보를 저장 및 조회가 가능한 시스템** 이다.

* 장기 메모리를 구현하기 위해서는 **LangGraph 의 영속성 (Persistence)** 기능을 사용한다.
* 장기 메모리에 저장되는 데이터는 **사용자에 대한 여러 가지 정보** 이다.
  * 사용자 프로필 정보
  * 사용자 선호도 데이터
  * 과거 사용자-LLM 간 상호작용 패턴 등
