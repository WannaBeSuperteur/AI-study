## 목차

* 목차
  * [1. LangChain의 메모리 개요 및 필요성](#1-langchain의-메모리-개요-및-필요성)
  * [2. LangChain의 메모리 유형](#2-langchain의-메모리-유형)
  * [3. 메모리의 구현 방식](#3-메모리의-구현-방식)
    * [3-1. RunnableWithMessageHistory](#3-1-runnablewithmessagehistory)
    * [3-2. LangGraph 기반 방법](#3-2-langgraph-기반-방법)
  * [4. 실전 예제](#4-실전-예제)
    * [4-1. 단기 메모리](#4-1-단기-메모리)
    * [4-2. 장기 메모리 (+ 문제점 및 원인 추정)](#4-2-장기-메모리)
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

**실전 예제 코드**

* OpenAI LLM 환경 설정

```python
# setting OpenAI API

from langchain_openai import ChatOpenAI
import os

with open('openai_api_key.txt', 'r') as f:
    openai_api_key = f.readlines()[0].split('\n')[0]
    os.environ['OPENAI_API_KEY'] = openai_api_key

llm = ChatOpenAI(model='gpt-4o-mini')
```

```python
from langgraph.store.memory import InMemoryStore

# 메모리 기반 Store 생성
store = InMemoryStore()
```

```python
# store 에 저장하기 위한 key 이름

USER_INFO_KEY = 'user_info'
```

* LLM 에이전트의 tool call 용 함수들

```python
from langchain.tools import tool, ToolRuntime

# 사용자 정보 조회

@tool
def get_user_info(info_type: str, runtime: ToolRuntime) -> str:
    """Get user info of info_type."""

    # get user ID
    store = runtime.store
    user_id = runtime.config['metadata'].get("user_id", "default")

    # search user_info (dict-like)
    namespace = ("users", user_id)
    memory = store.get(namespace, USER_INFO_KEY)

    if memory:
        try:
            return f"사용자의 {info_type} 정보: {memory.value[info_type]}"
        except:
            return f"{info_type} 과 일치하는 데이터가 없습니다."
    else:
        return "저장된 데이터가 없습니다."
```

```python
# 사용자 정보 저장 (기록)

@tool
def set_user_info(info_type: str, info_value: str, runtime: ToolRuntime) -> str:
    """Store user info of info_type as info_value."""

    # get user ID
    store = runtime.store
    user_id = runtime.config['metadata'].get("user_id", "default")

    # search or create user_info (dict-like)
    namespace = ("users", user_id)
    memory = store.get(namespace, USER_INFO_KEY)

    if memory:
        user_info = memory.value
    else:
        user_info = {}

    # store user info
    user_info[info_type] = info_value
    store.put(namespace, USER_INFO_KEY, user_info)

    return f"{info_type} 정보가 업데이트되었습니다."
```

* LLM 에이전트 실행

```python
# LLM 에이전트 생성

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    llm,
    tools=[get_user_info, set_user_info],
    checkpointer=MemorySaver(),
    store=store
)
```

```python
# LLM 에이전트 실행 (정보 저장 - 1)

config = {
    "configurable": {"thread_id": "thread-1"},
    "user_id": "test"
}
result = agent.invoke(
    {"messages": [("user", "내 경력을 3년 5개월로 저장해 줘")]},
    config=config
)

# LLM 실행 결과 출력
result['messages'][-1].content
```

* 결과
  * ```경력이 3년 5개월로 저장되었습니다. 다른 도움이 필요하시면 말씀해 주세요!```

```python
# LLM 에이전트 실행 (정보 저장 - 2)

config = {
    "configurable": {"thread_id": "thread-1"},
    "user_id": "test"
}
result = agent.invoke(
    {"messages": [("user", "내 전공분야를 머신러닝/딥러닝으로 저장해 줘")]},
    config=config
)

# LLM 실행 결과 출력
result['messages'][-1].content
```

* 결과
  * ```전공분야가 머신러닝/딥러닝으로 저장되었습니다. 추가로 필요한 사항이 있으면 알려주세요!```

```python
# LLM 에이전트 실행 (정보 저장 - 3)

config = {
    "configurable": {"thread_id": "thread-1"},
    "user_id": "test"
}
result = agent.invoke(
    {"messages": [("user", "내 학력을 석사 졸업으로 저장해 줘")]},
    config=config
)

# LLM 실행 결과 출력
result['messages'][-1].content
```

* 결과
  * ```학력이 석사 졸업으로 저장되었습니다. 더 필요한 것이 있으면 말씀해 주세요!```

```python
# LLM 에이전트 실행 (정보 조회)

config = {
    "configurable": {"thread_id": "thread-2"},
    "user_id": "test"
}
result = agent.invoke(
    {"messages": [("user", "내 경력 정보를 조회해 줘")]},
    config=config
)

# LLM 실행 결과 출력
result['messages'][-1].content
```

* 결과
  * ```당신의 경력 정보는 3년 5개월입니다. 추가적으로 필요한 정보가 있으면 말씀해 주세요!``` 

```python
# 모든 정보 조회
def get_all_user_info(store, user_id):

    # search user_info (dict-like)
    namespace = ("users", user_id)
    memory = store.get(namespace, USER_INFO_KEY)
    print(memory.value)
```

```python
get_all_user_info(store, user_id='test')
```

* 결과
  * ```{'경력': '3년 5개월', '전공분야': '머신러닝/딥러닝', '학력': '석사 졸업'}``` 

```python
# LLM 에이전트 실행 (추가 질문)

config = {
    "configurable": {"thread_id": "thread-3"},
    "user_id": "test"
}
result = agent.invoke(
    {"messages": [("user", "지금까지 저장된 전공분야, 학력, 경력 정보를 바탕으로, 나에게 맞는 이직 준비 전략을 추천해 줘")]},
    config=config
)

# LLM 실행 결과 출력
result['messages'][-1].content
```

* 결과

```
당신의 전공 분야, 학력, 경력 정보를 바탕으로 다음과 같은 이직 준비 전략을 추천합니다.

### 1. 전공 분야
- **전공 정보가 없음**: 전공이 명확하지 않으므로, 이직을 고려할 때 다양한 분야에 도전할 수 있습니다. 관심 있는 분야를 탐색하고, 해당 분야에 맞는 경로를 설정하는 것이 중요합니다.

### 2. 학력
- **산업 관련 전문학교 졸업**: 이 학벌은 여러 산업 분야에서 기초적인 전문 지식을 갖추고 있으므로, 이를 살릴 수 있는 분야를 정하는 것이 좋습니다.

### 3. 경력
- **3년 5개월의 경력**: 중간 경력자로서, 이직 시 경력을 강조하여 더 높은 직급이나 책임 있는 역할을 요청할 수 있습니다.

---

### 이직 준비 전략

1. **관심 분야 발굴**:
   - 어떤 분야에서 일하고 싶은지 깊이 고민해 보세요. IT, 마케팅, 디자인, 운영관리 등 다양한 옵션을 고려할 수 있습니다.

2. **기술 향상**:
   - 현재 산업이나 희망 직무에 필요한 추가 교육이나 자격증 취득을 고려해 보세요. 신기술이나 트렌드를 반영한 교육을 통해 경쟁력을 높이는 것이 중요합니다.

3. **이력서 업데이트**:
   - 전문학교에서의 경험, 프로젝트, 성과 등을 중심으로 이력서를 업데이트하세요. 경력과 관련된 구체적인 성과를 강조하는 것이 좋습니다.

4. **네트워킹 및 커뮤니티 참여**:
   - 관련 업종의 행사, 세미나, 온라인 커뮤니티에 적극 참여하여 인맥을 넓히고 정보 교류를 하세요. 네트워킹은 이직에 큰 도움이 됩니다.

5. **면접 준비**:
   - 목표로 하는 직무에 대한 충분한 연구 후, 관련 질문을 미리 준비하세요. 차별화된 답변을 준비하여 면접 시 자신감을 가지고 대처하는 것이 필요합니다.

6. **모의 면접 실시**:
   - 친구나 멘토를 통해 모의 면접을 진행하여 실제 면접 상황에 대비하세요. 이 과정을 통해 피드백을 받고 개선할 수 있습니다.

이직은 개인의 경력을 발전시키는 중요한 기회입니다. 충분한 준비를 통해 원하는 방향으로 나아가길 바랍니다. 추가적인 질문이나 도움이 필요하시면 언제든지 말씀해 주세요!
```

**문제점 및 원인 추정**

* 문제점
  * 전공 분야, 학력, 경력 정보가 불완전하게 추출됨
* 추정 원인
  * 사용자 정보를 저장한 dictionary 의 key 값이, 매번 LLM을 호출할 때마다 서로 다름
  * 이로 인해, **정보 저장 시의 key 값과 정보 호출 시의 key 값이 서로 불일치** 할 수 있음
* 고려해 볼 만한 해결 방법
  * 정보 저장 및 호출 함수에서 **지정된 key 값만을 사용** 하도록 프롬프팅
    * 정보의 종류가 한정되어 있지 않은 이상, 현실적으로 cost 가 클 것으로 보임

