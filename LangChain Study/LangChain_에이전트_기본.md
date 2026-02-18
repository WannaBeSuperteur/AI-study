## 목차

* 목차
  * [1. LangChain 의 Agent 개요](#1-langchain-의-agent-개요)
  * [2. Agent 기본 생성 및 실행 방법](#2-agent-기본-생성-및-실행-방법)
  * [3. Agent 의 구성 요소](#3-agent-의-구성-요소)
    * [3-1. LLM (언어 모델)](#3-1-llm-언어-모델)
    * [3-2. Tool (도구)](#3-2-tool-도구)
    * [3-3. 시스템 프롬프트](#3-3-시스템-프롬프트)
  * [4. 에이전트 실행 방법 상세](#4-에이전트-실행-방법-상세)
* ipynb 실습 파일
  * [ipynb 실습 파일](ipynb/LangChain_에이전트_기본.ipynb) 

## 1. LangChain 의 Agent 개요

**Agent (에이전트)** 는 거대 언어 모델 (LLM) 과 도구 (tools) 가 결합된 시스템이다.

* LLM 자체와 다른, Agent만의 특징은 다음과 같다.
  * 사용자 질문 분석을 통해, **어떤 도구를 사용할지 자율적으로 결정**
  * 도구 실행을 통해 얻은 결과를 바탕으로 **추가 작업 수행 여부 판단**

![image](images/LangChain_에이전트_1.PNG)

## 2. Agent 기본 생성 및 실행 방법

* LLM Agent 기본 생성 방법

```python
from langchain.agents import create_agent

# create agent
llm = ChatOpenAI(model='gpt-4o-mini')

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="당신은 고등학교 수학 교사입니다."
)
```

* LLM Agent 기본 실행 방법

```python
# execute agent

result = agent.invoke({
    "messages": [{"role": "user", "content": "벡터의 내적에 대해 설명해줘"}]
})
result['messages'][-1].content
```

* 실행 결과

```
벡터의 내적(또는 점곱, 스칼라 곱)은 두 벡터 사이의 연산으로, 그 결과는 스칼라(하나의 숫자)입니다. 내적은 주로 두 벡터의 유사성이나 방향을 비교하는 데 사용됩니다. 

두 벡터 \( \mathbf{a} = (a_1, a_2, ..., a_n) \)와 \( \mathbf{b} = (b_1, b_2, ..., b_n) \)의 내적은 다음과 같이 정의됩니다:

\[
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \ldots + a_n b_n
\]

즉, 각 벡터의 대응하는 성분을 곱한 후, 그 결과를 모두 더하는 방식입니다.

### 기하학적 의미
내적은 또한 두 벡터의 각도와 관련이 있습니다. 두 벡터 \( \mathbf{a} \)와 \( \mathbf{b} \)의 내적은 다음과 같은 공식으로도 표현할 수 있습니다:

\[
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)
\]

여기서 \( \|\mathbf{a}\| \)와 \( \|\mathbf{b}\| \)는 각각 벡터 \( \mathbf{a} \)와 \( \mathbf{b} \)의 크기(길이)이고, \( \theta \)는 두 벡터 사이의 각도입니다. 이 식에서 알 수 있듯이, 두 벡터가 서로 수직일 경우(즉, \( \theta = 90^\circ \)), 내적은 0이 됩니다.

### 내적의 성질
1. **교환법칙**: \( \mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a} \)
2. **결합법칙**: \( \mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c} \)
3. **배분법칙**: \( k(\mathbf{a} \cdot \mathbf{b}) = (k\mathbf{a}) \cdot \mathbf{b} \) (여기서 \( k \)는 스칼라입니다)
4. **양의 정의**: \( \mathbf{a} \cdot \mathbf{a} \geq 0 \)이며, \( \mathbf{a} \cdot \mathbf{a} = 0 \)일 때 \( \mathbf{a} \)는 영벡터입니다.

벡터 내적은 물리학, 컴퓨터 그래픽스, 기계 학습 등 다양한 분야에서 중요한 역할을 합니다. 이 개념을 잘 이해하면 벡터 간의 관계를 분석하는 데 많은 도움이 됩니다.
```

## 3. Agent 의 구성 요소

Agent의 구성 요소는 다음과 같다.

| 구성 요소       | 설명                                                           |
|-------------|--------------------------------------------------------------|
| LLM (언어 모델) | Agent의 **핵심적인 두뇌 역할 (AI 역할)**                                |
| Tool (도구)   | Agent가 **실제 작업 (인터넷 검색, 코드 실행, DB 쿼리 등)** 을 할 수 있게 하는 기능 제공자 |
| 시스템 프롬프트    | Agent의 **역할 및 작동 방식에 대한 지침**                                 |

### 3-1. LLM (언어 모델)

**LLM (언어 모델)** 은 LLM Agent의 두뇌, 즉 AI 역할이다.

* 모델 설정 방법

| 모델 설정 방법 | 설명                                               |
|----------|--------------------------------------------------|
| 정적 모델 설정 | LLM Agent 생성 시 **모델을 고정적으로 지정 (예: GPT-4o-mini)** |
| 동적 모델 선택 | 상태, Context 등을 고려하여 **동적으로 모델 선택**               |

* 동적 모델 선택 방법
  * 모델을 동적으로 선택하는 함수에서는 ```@wrap_model_call``` 미들웨어를 사용
  * **API 토큰 비용 최적화** 등에 활용 가능
    * 간단한 질문: ```gpt-4o-mini``` (API 비용이 저렴한 저성능 모델)
    * 복잡한 질문: ```gpt-4o``` (고성능 모델)

### 3-2. Tool (도구)

**Tool (도구)** 은 Agent가 실제로 작업을 수행할 수 있게 하는 기능을 지원한다.

* Tool을 이용한 작업 예시
  * 웹 검색 또는 DB로부터 쿼리 호출
  * 코드 실행, 수학 계산
  * 외부 API 호출
* Agent 가 Tool을 호출하는 메커니즘
  * 여러 도구의 **순차적** 호출하여, 이전 도구 출력을 다음 도구 입력으로 사용
  * **독립적으로 실행 가능** 한 도구들을 **병렬적** 호출
  * 도구 실행 결과값을 **상태에 저장** 하여, 향후 활용 가능하게 함

### 3-3. 시스템 프롬프트

**시스템 프롬프트 (System Prompt)** 는 Agent의 역할 및 Agent의 작동 방식에 대한 지침을 텍스트로 나타낸 것이다.

* 시스템 프롬프트 작성 팁
  * Agent의 **역할에 대한 명확한 기술**
  * **작업 방식** 및 **제약 조건 (지켜야 할 형식, 하지 말아야 할 것 등)** 안내

## 4. 에이전트 실행 방법 상세

**1. 단일 실행 (```invoke()```)**

* [2. Agent 기본 생성 및 실행 방법](#2-agent-기본-생성-및-실행-방법) 의 예제 참고.

**2. 스트리밍 실행 (```stream()```)**

* 스트리밍 실행 모드

| ```updates```               | ```messages``` | ```custom``` |
|-----------------------------|----------------|--------------|
| 각 에이전트 **단계별 결과 도출** 시 업데이트 | **토큰 단위** 스트리밍 | 사용자 지정 스트리밍  |

* 예제 코드

```python
# 스트리밍 방식

for event in agent.stream(
    {"messages": [{"role": "user", "content": "벡터의 내적이 갖는 의미를 20글자 이내의 명사 조합으로 설명해줘."}]},
    stream_mode="messages"
):
    print(event)
```

* 실행 결과

```
(AIMessageChunk(content='', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='두', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content=' 벡', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='터', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='의', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content=' 유', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='사', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='성', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content=' 측', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='정', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='.', additional_kwargs={}, response_metadata={'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_373a14eb6f', 'service_tier': 'default', 'model_provider': 'openai'}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 50, 'output_tokens': 11, 'total_tokens': 61, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}, tool_call_chunks=[]), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
(AIMessageChunk(content='', additional_kwargs={}, response_metadata={}, id='lc_run--019c6fc0-92c6-7973-826c-d4afdb1a06f6', tool_calls=[], invalid_tool_calls=[], tool_call_chunks=[], chunk_position='last'), {'langgraph_step': 1, 'langgraph_node': 'model', 'langgraph_triggers': ('branch:to:model',), 'langgraph_path': ('__pregel_pull', 'model'), 'langgraph_checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'checkpoint_ns': 'model:9fb5d23b-bfab-f612-ebcf-4c041d83a34e', 'ls_provider': 'openai', 'ls_model_name': 'gpt-4o-mini', 'ls_model_type': 'chat', 'ls_temperature': None})
```

**3. 대화 히스토리 관리**

여러 턴의 대화가 이어지는 동안 메시지 기록을 유지한다.

* 예제 코드

```python
# 대화 히스토리 관리

from langchain_core.messages import HumanMessage

# initialize message history
messages = []
messages.append(HumanMessage(content="다항함수의 미분법을 알려줘"))
result = agent.invoke({"messages": messages})
print('\n\nLLM 답변 (1):\n', result["messages"][-1].content)

# add result to message history
messages.extend(result["messages"])

# additional question with context
messages.append(HumanMessage(content="그럼 이걸로 간단한 수학 문제 하나 내줘"))
result = agent.invoke({"messages": messages})
print('\n\n\n\nLLM 답변 (2):\n', result["messages"][-1].content)
```

* 실행 결과

```
LLM 답변 (1):
 다항함수의 미분법은 매우 간단하며, 기본적인 미분 규칙을 활용합니다. 다항함수는 일반적으로 다음과 같은 형태로 표현됩니다.

\[ f(x) = a_n x^n + a_{n-1} x^{n-1} + \ldots + a_1 x + a_0 \]

여기서 \( a_n, a_{n-1}, \ldots, a_1, a_0 \)는 상수이고, \( n \)은 자연수입니다.

다항함수의 미분법의 기본 원칙은 다음과 같습니다:

1. **거듭제곱 미분법**: 
   \( x^n \)의 미분은 다음과 같습니다.
   \[ \frac{d}{dx}(x^n) = n x^{n-1} \]

2. **상수 미분법**: 
   상수의 미분은 0입니다.
   \[ \frac{d}{dx}(c) = 0 \] (여기서 \( c \)는 상수)

3. **선형성**: 
   미분은 합과 차에 대해 선형적입니다.
   \[ \frac{d}{dx}(f(x) + g(x)) = \frac{d}{dx}(f(x)) + \frac{d}{dx}(g(x)) \]
   \[ \frac{d}{dx}(cf(x)) = c \cdot \frac{d}{dx}(f(x)) \] (여기서 \( c \)는 상수)

위의 원칙을 사용하여 다항함수를 미분할 수 있습니다. 예를 들어, 다음 다항함수를 미분해 보겠습니다.

\[ f(x) = 3x^4 + 5x^3 - 2x + 7 \]

이 함수의 미분은 다음과 같이 수행됩니다:

1. \( 3x^4 \)의 미분: \( 3 \cdot 4 x^{4-1} = 12x^3 \)
2. \( 5x^3 \)의 미분: \( 5 \cdot 3 x^{3-1} = 15x^2 \)
3. \( -2x \)의 미분: \( -2 \)
4. \( 7 \)의 미분: \( 0 \)

따라서, 전체 미분 결과는 다음과 같습니다:

\[ f'(x) = 12x^3 + 15x^2 - 2 \]

이와 같은 방식으로 다항함수를 미분할 수 있습니다. 추가적으로 궁금한 내용이나 설명이 필요한 부분이 있다면 말씀해 주세요!
```

```
LLM 답변 (2):
 좋습니다! 다음은 다항함수의 미분을 이용한 간단한 문제입니다.

**문제:** 

다음 다항함수 \( f(x) = 2x^5 - 3x^4 + 4x^2 - 7 \)의 미분을 구하고, \( x = 1 \)에서의 기울기를 구하세요.

**풀이 방법:**
1. 먼저 \( f(x) \)를 미분합니다.
2. 그 다음, 미분한 결과를 \( x = 1 \)에 대입하여 기울기를 구합니다.

과제를 완료한 후에 답을 확인해드릴게요!
```
