## 목차

* [1. 미들웨어의 개요 및 역할](#1-미들웨어의-개요-및-역할)
  * [1-1. 미들웨어의 hook 종류](#1-1-미들웨어의-hook-종류)
  * [1-2. 미들웨어의 기본 사용법](#1-2-미들웨어의-기본-사용법)
  * [1-3. 미들웨어의 실행 순서](#1-3-미들웨어의-실행-순서)
* [2. LangChain 내장 미들웨어](#2-langchain-내장-미들웨어)
  * [2-1. 가드레일](#2-1-가드레일) 

## 1. 미들웨어의 개요 및 역할

LangChain에서 **미들웨어 (Middleware)** 는 **LangChain 파이프라인 사이에 끼워넣는 실행 함수 (hook)** 이다.

* 미들웨어의 역할은 다음과 같다.
  * LLM 에이전트의 동작 추적 (로깅, 디버깅, 실행 시간 측정 등)
  * 프롬프트 변환
  * 실패 시 fallback 처리
  * 가드레일 (호출 제한 등)

즉, 일명 **cross-cutting concerns (횡단 관심사)** 를 처리하는 역할이다.

### 1-1. 미들웨어의 hook 종류

미들웨어 훅 (hook) 의 종류는 **Node-style** 훅과 **Wrap-style** 훅이 있다.

* Node-style 훅
  * **특정 지점에서 순차적으로** 실행된다.

| ```before_agent```       | ```before_model```         | ```after_model```          | ```after_agent```     |
|--------------------------|----------------------------|----------------------------|-----------------------|
| 에이전트 동작 시작 전<br>**(1회)** | 모델 호출 전<br>(모델 **호출 시마다**) | 모델 응답 후<br>(모델 **호출 시마다**) | 에이전트 종료 후<br>**(1회)** |

* Wrap-style 훅
  * **모델/도구 호출 부분을 '감싸는'** 형태로 실행된다.
  * fallback 처리 및 모델/도구 실행 재시도 등에 사용할 수 있다. 

| ```wrap_model_call``` | ```wrap_tool_call``` |
|-----------------------|----------------------|
| **모델 호출** 을 감싸는 hook  | **도구 호출** 을 감싸는 hook |

### 1-2. 미들웨어의 기본 사용법

미들웨어는 다음과 같은 형태로 사용한다.

* 여기서 ```hook_name``` 에는 hook의 이름 (```before_agent``` 등) 이 들어간다.

```python
# hook 함수 정의

@hook_name
def hook_function(state: AgentState, runtime: Runtime):
    """function description"""

    # function body code

    return None
```

```python
# 에이전트 생성

agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[hook_function, ...]  # 사용할 middleware 함수의 리스트
)
```

### 1-3. 미들웨어의 실행 순서

각 미들웨어의 실행 순서는 다음과 같다.

| 미들웨어                                       | 실행 순서                 |
|--------------------------------------------|-----------------------|
| ```before_model``` ```before_agent```      | 미들웨어 list **순서대로** 실행 |
| ```after_model``` ```after_agent```        | 미들웨어 list **역순으로** 실행 |
| ```wrap_model_call``` ```wrap_tool_call``` | **중첩되어** 실행           |

## 2. LangChain 내장 미들웨어

LangChain에서 제공하는 내장 미들웨어는 다음과 같이 10가지이다.

| 미들웨어 카테고리   | 미들웨어                                                                                                                                                               |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 컨텍스트 관리     | - ```SummarizationMiddleware``` (자동 요약)<br>- ```ContextEditingMiddleware``` (컨텍스트 수정을 통해 **오래된 도구 호출 결과 제거**)                                                      |
| 호출 제한       | - ```ModelCallLimitMiddleware``` (모델 **API 호출** 횟수 제한)<br>- ```ToolCallLimitMiddleware``` (**도구 호출** 횟수 제한)                                                        |
| 오류 대응 및 재시도 | - ```ModelFallbackMiddleware``` (모델 실패, 즉 fallback 시 **대체 모델로 전환**)<br>- ```ModelRetryMiddleware``` (**모델 호출** 재시도)<br>- ```ToolRetryMiddleware``` (**도구 호출** 재시도) |
| 도구 최적화      | - ```LLMToolSelectorMiddleware``` (도구 선택)                                                                                                                          |
| 가드레일        | - ```PIIMiddleware``` (**개인정보** 탐지 및 마스킹 처리)<br>- ```HumanInTheLoopMiddleware``` (실행 시 **사전에 사람의 승인** 필요)                                                          |

### 2-1. 가드레일

LangChain 미들웨어 중 **가드레일 (Guardrails)** 은 **AI 에이전트 실행 시 안전성, 품질 등을 보장** 하기 위한 장치이다.

* 미들웨어의 역할은 다음과 같다.
  * 개인정보 등 **민감한 정보 유출 방지**
  * 각종 악의적 공격 (prompt injection 등), 부적절한 콘텐츠 등 차단
  * 각 산업별 AI Agent가 지켜야 하는 **규정 준수**
  * AI Agent 출력에 대한 **품질 및 정확성 보증**

가드레일 구현 방법은 **결정적 (Deterministic), 모델 기반 (Model-based)** 가드레일이 있다.

| 구현 방법               | 설명                                             |
|---------------------|------------------------------------------------|
| 결정적 (Deterministic) | **규칙 기반 로직** (정규 표현식 등) 을 통한 가드레일 구현           |
| 모델 기반 (Model-based) | **AI 모델 (LLM, 분류 모델 등)** 을 사용한 **의미론적 콘텐츠 평가** |

* 실제 구현 예시

```python
@before_agent
def validate_user_input(state: AgentState, runtime: Runtime):
    """Validate user input prompt."""

    # ... function body ...
    
    if not is_valid:  # raise ValueError on non-valid user input
        raise ValueError(...)
```
