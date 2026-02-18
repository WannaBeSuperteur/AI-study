## 목차

* 목차
  * [1. LangChain에서 호출 가능한 도구](#1-langchain에서-호출-가능한-도구)
    * [1-1. 내장 도구](#1-1-내장-도구)
    * [1-2. 커스텀 도구](#1-2-커스텀-도구)
  * [2. Agent와 도구 통합 예제](#2-agent와-도구-통합-예제)
  * [3. 커스텀 도구를 활용한 에이전트 생성 예제](#3-커스텀-도구를-활용한-에이전트-생성-예제)
    * [3-1. 문제점 및 해결 방법](#3-1-문제점-및-해결-방법)
    * [3-2. 참고 사항](#3-2-참고-사항)
* ipynb 실습 파일
  * [도구 호출 기본 (Agent와 도구 통합 예제)](ipynb/LangChain_에이전트_도구_호출_기본.ipynb)
  * [커스텀 도구를 활용한 에이전트 생성 예제](ipynb/LangChain_에이전트_도구_호출_커스텀.ipynb)

## 1. LangChain에서 호출 가능한 도구

LangChain에서 호출 가능한 도구 (tool) 는 **내장 도구** 와 **커스텀 도구** 로 구분된다.

| 구분     | 설명                                                                    | 예시                  |
|--------|-----------------------------------------------------------------------|---------------------|
| 내장 도구  | LangChain에서 자체적으로 제공하는 도구                                             | 웹 검색 도구, 코드 인터프리터 등 |
| 커스텀 도구 | **Python 함수 형태** 의 도구<br>- 커스텀 도구로 사용할 함수에 ```@tool``` 데코레이터를 이용하여 생성 | -                   |

### 1-1. 내장 도구

LangChain의 내장 도구의 종류는 다음과 같다.

| 내장 도구 종류  | 설명 및 예시                                       | 추가 설명                   |
|-----------|-----------------------------------------------|-------------------------|
| 웹 검색 도구   | Tavily Search, Google Search 등                | Brave Search는 무료로 사용 가능 |
| 코드 인터프리터  | 코드 실행 환경 제공 (PythonAstREPLTool 등)             | 주로 Python 언어를 지원        |
| 생산성 도구    | GitHub/Gmail/Slack Toolkit                    | -                       |
| 웹 브라우징 도구 | 웹 브라우저에서의 **작업을 자동화** (브라우저와 상호작용, HTTP 요청 등) | -                       |
| 데이터베이스 도구 | DB 작업 자동화 (SQLDatabase Toolkit 등)             | -                       |

### 1-2. 커스텀 도구

커스텀 도구는 **Python 함수 형태로 작성** 하는 도구이다.

* 다음과 같이 커스텀 도구로 사용할 함수에 ```@tool``` 데코레이터를 사용한다.

```python
@tool
def function_name(arg: ...) -> str:
    """function description
    ...
    """

    # processing function code here

    return final_result
```

## 2. Agent와 도구 통합 예제

**1. 기본 통합**

여기서는 **웹 검색** 도구를 활용한다.

* 예제 코드

```python
# 기본 통합

from langchain.agents import create_agent
from langchain_community.tools import TavilySearchResults

# 검색 도구 초기화
search_tool = TavilySearchResults(max_results=10)

# 에이전트 생성
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="""당신은 웹 검색을 통해 정보를 제공하는 어시스턴트입니다.
사용자의 질문에 대해 신뢰성 있는 출처의 정보를 찾아서 답변을 검색합니다."""
)

# 에이전트 실행
user_query = "각 업체 (OpenAI, Google, Anthropic) 별 2025년 4분기 이후 출시된 거대 언어 모델들의 성능을 비교 분석해줘"
result = agent.invoke({
    "messages": [{"role": "user", "content": user_query}]
})

print(result["messages"][-1].content)
```

* 실행 결과

```
아래는 OpenAI, Google, Anthropic의 2025년 4분기 이후 출시된 거대 언어 모델들의 성능을 비교 분석한 내용입니다.

### OpenAI (GPT-5 시리즈)
- **모델**: GPT-5, GPT-5.1, GPT-5.2
- **주요 성능 지표**:
  - **코드 작성 능력**: SWE-bench Verified 기준으로 GPT-5는 74.9%의 정확도를 기록하여 상대적으로 높은 성능을 보였습니다. 통합된 Python 기능을 이용한 GPT-5 Pro는 89.4%의 정확도를 기록하여 뛰어난 성능을 보였습니다.
  - **수학적 성능**: AIME 2025 기준 94.6%로, 이는 Gemini 3 Pro에 이어 두 번째로 우수한 성적입니다.
- **기타 특징**: 
  - **비용**: API 사용 가격은 상당히 높은 편이며, $3에서 $12 per 1M tokens 사이입니다.
  - **창의성 및 안정성**: GPT-5는 이전 모델에 비해 오류가 80% 감소하였으며, 더 나은 확신 평가 기능을 갖추었습니다.

### Google (Gemini 2.5 Pro 및 Gemini 2.5 Flash)
- **모델**: Gemini 2.5 Pro, Gemini 2.5 Flash
- **주요 성능 지표**:
  - **코드 작성 능력**: SWE-bench에서 76.2%의 정확도를 기록하여 OpenAI의 GPT-5보다 낮지만, 가격 대비 성능 비율에서 큰 장점을 보입니다.
  - **수학적 성능**: AIME 시험에서 82%의 성능을 기록했습니다.
- **기타 특징**:
  - **비용**: Gemini 2.5 Pro는 $1.25 per 1M tokens의 입력 비용 및 $10 per 1M tokens의 출력 비용을 청구합니다. Flash 모델은 더 저렴한 옵션으로 제공됩니다.
  - **속도와 멀티모달 처리**: Gemini 모델은 속도와 멀티모달 처리에서 우수한 성능을 보이며, 다양한 응용 프로그램에 적합합니다.

### Anthropic (Claude Opus 4.5)
- **모델**: Claude Opus 4.5, Claude Sonnet 4.5
- **주요 성능 지표**:
  - **코드 작성 능력**: Claude Opus 4.5는 SWE-bench Verified에서 80.9%의 최고의 성능을 기록하였으며 이는 다른 경쟁 모델들을 초월합니다.
  - **수학적 성능**: 측정된 수학적 성능은 88%로 다른 모델들에 비해 상대적으로 높은 편입니다.
- **기타 특징**:
  - **비용**: API 비용은 OpenAI 모델에 비해 상대적으로 저렴하여, 대규모 프로젝트에도 유리합니다.
  - **윤리적 AI**: Claude는 윤리적 고려가 돋보이는 설계로 인해 교육, 의료, 법률 분야에서의 활용성이 뛰어납니다.

### 성능 요약
- **코딩 성능**: Claude Opus 4.5 > GPT-5 > Gemini 2.5 Pro
- **수학적 성능**: Gemini 3 Pro > GPT-5.1 > Claude Opus 4.5
- **비용 효율성**: Gemini 2.5 Flash 모델은 가장 저렴한 사용 비용을 가집니다.

각 모델은 특정 작업에 대해 최적화되어 있으며, 선택은 사용자의 요구에 따라 달라질 수 있습니다. OpenAI의 모델은 창의성과 통합된 기능에서 우수하며, Google 모델은 비용과 성능 비율에서, Anthropic의 모델은 윤리적 고려와 고급 코딩 기능에서 강점을 보입니다.
```

**2. 여러 가지 도구를 조합하여 통합**

여기서는 **웹 검색** 도구와 **Python 코드 인터프리터** 를 조합하여 활용한다.

* 예제 코드

```python
# 여러 가지 도구 조합하여 통합

from langchain_experimental.tools import PythonAstREPLTool

# initialize tools
search_tool = TavilySearchResults(max_results=5)
python_tool = PythonAstREPLTool()

# create multiple-tools LLM agent
agent = create_agent(
    model=llm,
    tools=[search_tool, python_tool],
    system_prompt="""당신은 역량 있는 데이터 분석가입니다.
- 관련 정보가 필요하면 웹 검색을 실시합니다.
- 계산 또는 데이터 처리가 필요하면 Python 코드를 작성하고 실행합니다."""
)

# 복합 작업 실행
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """비트코인의 최근 5년간 연 평균 수익률을 계산해줘.
- 비트코인의 최근 5년치 가격을 먼저 검색한다.
- 이 가격을 바탕으로 수학 공식을 이용하여 연평균 수익률을 계산한다.
        """
    }]
})

print(result["messages"][-1].content)
```

* 실행 결과

```
비트코인의 최근 5년간 연 평균 수익률은 약 **11.41%**입니다.
```

## 3. 커스텀 도구를 활용한 에이전트 생성 예제

* 예제 코드

```python
# LLM이 호출할 도구 (tool) 로 사용할 함수 (evaluate_expression)

from langchain.tools import tool

@tool
def evaluate_expression(expression: str) -> str:
    """Evaluate given math expression."""

    print(f'\n\n[ expression to solve ] {expression}\n\n')

    python_expression = expression.replace('x', '*')
    return eval(python_expression)
```

```python
from langchain.agents import create_agent

# create LLM agent
agent = create_agent(
    model=llm,
    tools=[evaluate_expression],
    system_prompt="""당신은 수학 교사입니다.
- 사용자가 수학 문제를 제시하면, 해당 문제에 대한 풀이 과정 및 답을 제시합니다.
- 사용자가 제시하는 수학 문제는 실생활 기반 문장제 문제입니다."""
)
```

```python
# run LLM agent (1)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": """
영준이는 사과 2개를 가지고 있고, 유영이는 사과 4개를 가지고 있다.
영준이와 유영이가 가진 사과의 개수의 합과 차는?        
"""}
    ]
})

print(result["messages"][-1].content)
```

```python
# run LLM agent (2)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": """
영준이가 가진 사과는 유영이가 가진 사과 개수의 3배이다.
영준이와 유영이가 각각 사과 8개씩을 더 가지면, 영준이가 가진 사과는 유영이가 가진 사과 개수의 2배가 된다.
영준이와 유영이가 각각 가진 사과의 개수는?       
"""}
    ]
})

print(result["messages"][-1].content)
```

* 실행 결과 (```run LLM agent (1)```, 일부 개행 제거)

```
[ expression to solve ] 2 + 4
[ expression to solve ] 4 - 2

영준이와 유영이가 가진 사과의 개수의 합은 6개, 차는 2개입니다. 

- 사과의 개수의 합: \(2 + 4 = 6\)
- 사과의 개수의 차: \(4 - 2 = 2\)
```

* 실행 결과 (```run LLM agent (2)```, 일부 개행 제거)

```
[ expression to solve ] x = 유영이의 사과 개수, 영준이의 사과 개수 = 3x (사과의 개수 관계)
[ expression to solve ] (3x + 8) = 2(x + 8) (사과 개수 조건)
[ expression to solve ] 3x + 8 = 2x + 16 (방정식 정리)

Traceback (most recent call last):

...

  File "<string>", line 1
    (3* + 8) = 2(* + 8) (사과 개수 조건)
             ^
SyntaxError: invalid syntax

```

### 3-1. 문제점 및 해결 방법

* 문제점
  * 주어진 tool 함수를 호출할 때 **입력 형식이 잘못된 경우 오류 발생 → 프로세스 종료됨**
* 해결 방법
  * tool 함수 호출에서 오류 발생 시, **오류가 발생했다는 예외 처리**
  * 오류가 발생했을 때 대안으로 호출해야 하는 tool을 별도로 준비
    * 해당 tool을 호출하도록 시스템 프롬프트 추가 구성 

### 3-2. 참고 사항

* tool 로 사용되는 함수의 인수명으로 ```config```, ```runtime``` 은 사용할 수 없다.
