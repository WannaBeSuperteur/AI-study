## 목차

* [1. LangChain의 모델 유형](#1-langchain의-모델-유형)
  * [1-1. LLM](#1-1-llm)
  * [1-2. Chat Model](#1-2-chat-model)
  * [1-3. 통합 모델 초기화](#1-3-통합-모델-초기화)
* [2. LLM 모델 파라미터 설정](#2-llm-모델-파라미터-설정)

## 1. LangChain의 모델 유형

LangChain에서 제공하는 모델의 유형은 다음과 같다.

| 모델 유형                             | 설명                                                           |
|-----------------------------------|--------------------------------------------------------------|
| LLM (거대 언어 모델)                    | OpenAI, HuggingFace 등 LLM 제공업체로부터 모델 사용 지원                   |
| Chat Model                        | **대화형 메시지** 를 입력하고, 마찬가지로 **대화형 메시지** 를 출력하는 특화 LLM          |
| 통합 모델 초기화 (```init_chat_model```) | 여러 LLM 제공자 (OpenAI, Google 등) 의 모델을 **통합된 인터페이스** 를 이용하여 초기화 |

### 1-1. LLM

**LLM (Large Language Models)** 인터페이스는 **OpenAI 등 다양한 업체에서 제공하는 LLM과의 상호작용** 을 지원한다.

* 즉, LangChain은 **LLM을 제공하는 것이 아닌, 다양한 업체의 LLM을 지원하는 플랫폼** 의 역할을 하는 것이다.

----

* 예시 코드 (OpenAI LLM 사용)

```python
from langchain_openai import OpenAI

llm = OpenAI()

llm.invoke("챗GPT야 안녕? 반가워!")
```

### 1-2. Chat Model

**Chat Model** 은 다음 특징을 갖는 **특화된 LLM 모델** 이다.

* 입력 메시지와 출력 메시지가 모두 **대화형 메시지**
* 일반 텍스트 대신 **대화의 맥락을 포함한 텍스트** 처리를 통해, **보다 자연스럽고 매력적인 대화** 가능

----

* 예시 코드 (OpenAI LLM 사용)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 20대 여성으로 설정된 당신의 친구 채지희입니다."),
    ("user", "{user_input}"),
])

chain = chat_prompt | chat
chain.invoke({"user_input": "챗GPT 지희야 안녕? 반가워!"})
```

### 1-3. 통합 모델 초기화

**통합 모델 초기화 (```init_chat_model```)** 는 여러 LLM 제공자 (OpenAI, Google 등) 의 모델을 **통합된 인터페이스** 를 이용하여 초기화하는 것을 말한다.

* LangChain 1.0에서는 ```init_chat_model``` 의 사용이 권장된다.

----

* 예시 코드
  * ```{openai_model_name}```, ```{anthropic_model_name}```, ```{gemini_model_name}``` 는 각각 해당 LLM 서비스 제공자의 모델 이름을 의미한다.

```python
from langchain.chat_models import init_chat_model

# OpenAI
openai_model = init_chat_model("gpt-{openai_model_name}")

# Anthropic Claude
claude_model = init_chat_model("claude-{anthropic_model_name}")

# Google Gemini
gemini_model = init_chat_model("google_genai:gemini-{gemini_model_name}")
```

## 2. LLM 모델 파라미터 설정

* LLM에는 ```temperature```, ```top-k```, ```top-p``` 와 같은 여러 파라미터들이 있다.
* 상세 설명
  * [LLM 기초 - LLM 디코딩 전략](../AI%20Basics/LLM%20Basics/LLM_기초_Decoding_Strategies.md) 참고
