## 목차

* 목차
  * [1. LLM 체인의 구성 요소](#1-llm-체인의-구성-요소)
  * [2. Chain 연결 및 호출 실습](#2-chain-연결-및-호출-실습)
    * [2-1. 기본 Chain 호출](#2-1-기본-chain-호출)
    * [2-2. Chain 연결 및 호출 (프롬프트 템플릿 이용)](#2-2-chain-연결-및-호출-프롬프트-템플릿-이용)
    * [2-3. Output Parser를 이용한 처리](#2-3-output-parser를-이용한-처리)
* ipynb 실습 파일
  * [ipynb 실습 파일](ipynb/LangChain_LLM_체인_기본.ipynb)

## 1. LLM 체인의 구성 요소

* **LLM 체인 (LLM chain)** 의 구성 요소는 다음과 같다.

| 구성 요소         | 설명                                                                |
|---------------|-------------------------------------------------------------------|
| 프롬프트 (prompt) | LLM (거대 언어 모델) 에 입력되는 프롬프트 (= LLM에 입력되는 지시문)                      |
| LLM           | 거대 언어 모델 (GPT, Gemini, Claude 등 상용 LLM + Transformers 자체 학습 모델 등) |

## 2. Chain 연결 및 호출 실습

### 2-1. 기본 Chain 호출

* 다음 형식으로 LLM을 호출하면 된다.

```python
llm_answer = llm.invoke(llm_prompt)
```

* Oh-LoRA LLM 호출 코드

```python
# 1. 기본 LLM Chain 실행

user_message = '로라야 너 친한 친구 한명 소개해줘'
final_llm_prompt = f'{user_message} (답변 시작)'
llm_answer = local_llm.invoke(final_llm_prompt)

# Oh-LoRA LLM의 실제 답변 부분
llm_answer.split('(답변 시작) ')[1].split('(답변 종료)')[0]
```

* 결과: ```내 제일 친한 친구 혜나! 👩 소개해 줄까?```

### 2-2. Chain 연결 및 호출 (프롬프트 템플릿 이용)

* **프롬프트 템플릿** 은 LLM에 입력되는 프롬프트를 **템플릿화** 시킨 것이다.
  * 다음과 같이 템플릿에 들어갈 내용을 ```{내용}``` 형식으로 하여 ```ChatPromptTemplate``` 을 만들면 된다.

```python
prompt = ChatPromptTemplate.from_template("... {내용} ...")
```

* Oh-LoRA LLM 호출 코드

```python
# 2. 기본 LLM Chain 실행 (프롬프트 템플릿 연계)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("[오늘 일정: 설날] (지금은 월요일 오후) {user_message} (답변 시작)")
prompt
```

```python
chain = prompt | local_llm
chain
```

```python
llm_answer_chain = chain.invoke({"user_message": "로라야 오늘 무슨 날인지 알아?"})
llm_answer_chain
```

```python
# Oh-LoRA LLM의 실제 답변 부분
llm_answer_chain.split('(답변 시작) ')[1].split('(답변 종료)')[0]
```

* 결과: ```오 오늘 설날이네! 🎉 설날에는 맛집 가서 🍲 마음껏 먹어야지!```

### 2-3. Output Parser를 이용한 처리

* 기본 설명
  * Output Parser 는 일종의 **출력 파서** 이다.
  * Output Parser 를 이용하여 LLM의 답변을 문자열 등 형식으로 parsing 할 수 있다.

* Oh-LoRA LLM 호출 코드

```python
# 3. Output Parser 를 이용한 처리

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain_with_parser = prompt | local_llm | output_parser
chain_with_parser
```

```python
llm_answer_chain_with_parser = chain_with_parser.invoke({"user_message": "로라야 오늘 무슨 날인지 알아?"})
llm_answer_chain_with_parser
```

```python
# Oh-LoRA LLM의 실제 답변 부분
llm_answer_chain_with_parser.split('(답변 시작) ')[1].split('(답변 종료)')[0]
```

* 결과: ```음… 오늘 월요일이잖아! 그래서 월요일에만 느낄 수 있는 그게 있지!```
