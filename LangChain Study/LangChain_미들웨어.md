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

### 2-1. 가드레일