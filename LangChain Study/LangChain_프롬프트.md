## 목차

* 목차
  * [1. 프롬프트의 개념](#1-프롬프트의-개념)
  * [2. 프롬프트 템플릿](#2-프롬프트-템플릿)
    * [2-1. 기본 프롬프트 템플릿 (PromptTemplate)](#2-1-기본-프롬프트-템플릿-prompttemplate) 
    * [2-2. ChatPromptTemplate](#2-2-chatprompttemplate)
    * [2-3. FewShotPromptTemplate](#2-3-fewshotprompttemplate)
  * [3. Partial Prompt](#3-partial-prompt)
  * [4. 참고 링크](#4-참고-링크)
* ipynb 실습 파일
  * [ipynb 실습 파일](ipynb/LangChain_프롬프트.ipynb)

## 1. 프롬프트의 개념

**프롬프트 (Prompt)** 는 사용자와 LLM 간의 상호작용에서, **사용자가 LLM에게 입력하는 대화, 질문, 요청 등의 텍스트** 이다.

* LLM의 답변 결정에 큰 역할을 한다.

## 2. 프롬프트 템플릿

**프롬프트 템플릿 (Prompt Template)** 은 LangChain에서 제공하는, **프롬프트에 대한 변수를 포함한 템플릿** 이다.

| 프롬프트 템플릿 종류                  | 설명                                                              |
|------------------------------|-----------------------------------------------------------------|
| 기본 프롬프트 템플릿 (PromptTemplate) | ```PromptTemplate.from_template(...)``` 형태로 정의되는 가장 간단한 형태의 템플릿 |
| ChatPromptTemplate           | **대화형 프롬프트 템플릿** 으로, 튜플 목록, 딕셔너리 등 **비교적 복잡한 구성** 가능            |
| FewShotPromptTemplate        | 사용할 예제를 포함한 프롬프팅 가능                                             |

### 2-1. 기본 프롬프트 템플릿 (PromptTemplate)

**1. 문자열 템플릿**

* 예시 코드

```python
template_text = "{topic}의 {discussion_topic}에 대해 {how} 알려줘."
prompt_template = PromptTemplate.from_template(template_text)

filled_prompt = prompt_template.format(
    topic="거대 언어 모델",
    discussion_topic="윤리적 문제",
    how="아주 자세히"
)
filled_prompt
```

* 실행 결과

```
거대 언어 모델의 윤리적 문제에 대해 아주 자세히 알려줘.
```

**2. 프롬프트 템플릿 간 결합**

다음과 같이 **문자열 또는 프롬프트 템플릿 간 서로 결합** 시켜 사용할 수 있다.

* 예시 코드

```python
# 프롬프트 템플릿 간 결합

combined_prompt = (
    prompt_template
    + PromptTemplate.from_template("\n\n그리고 {additional_topic} 알려줘.")
    + PromptTemplate.from_template("\n\n이때 {limit}글자 이내로 설명해줘.")
)

combined_prompt
```

* 실행 결과

```
PromptTemplate(input_variables=['additional_topic', 'discussion_topic', 'how', 'limit', 'topic'], input_types={}, partial_variables={}, template='{topic}의 {discussion_topic}에 대해 {how} 알려줘.\n\n그리고 {additional_topic} 알려줘.\n\n이때 {limit}글자 이내로 설명해줘.')
```

### 2-2. ChatPromptTemplate

**ChatPromptTemplate (챗 프롬프트 템플릿)** 은 **대화형에 최적화된 템플릿** 이다.

* 메시지 입력을 1개가 아닌 여러 개 (tuple 리스트) 로 구성할 수 있다.
* 이때 각 메시지는 ```role``` 과 ```content``` 로 구성된다.

----

* 예시 코드

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 Oh-LoRA (오로라 👱‍♀️) 로, 20대 여성으로 설정된 가상 인간입니다."),
    ("user", "{user_message}")
])

messages = chat_prompt.format_messages(user_message="로라야 안녕? 요즘 뭐하고 지내?")
messages
```

* 실행 결과

```
[SystemMessage(content='이 시스템은 Oh-LoRA (오로라 👱\u200d♀️) 로, 20대 여성으로 설정된 가상 인간입니다.', additional_kwargs={}, response_metadata={}),
 HumanMessage(content='로라야 안녕? 요즘 뭐하고 지내?', additional_kwargs={}, response_metadata={})]
```

### 2-3. FewShotPromptTemplate

**FewShotPromptTemplate** 은 언어 모델이 적절하게 답변할 수 있도록 **예제 (example) 를 제공하는 방식** 의 템플릿이다.

* 참고: [Few Shot Prompting](../AI%20Basics/LLM%20Basics/LLM_기초_Prompt_Engineering.md#3-2-few-shot-prompting)

----

* 예시 코드

```python
# 1. example formatter 구성

example_prompt = PromptTemplate.from_template("질문/답변: {question}\n{answer}")
```

```python
# 2. 예시 구성

examples = [
    {
        "question": "2026년 1월 1일로부터 10일이 지난 날은?",
        "answer": "20260111"
    },
    {
        "question": "2026년 크리스마스는 언제일까요?",
        "answer": "20261225"
    },
    {
        "question": "내가 LangChain의 매력에 푹 빠지기 시작한 날은?",
        "answer": "20260215"
    }
]
```

```python
# FewShotPromptTemplate 생성

from langchain_core.prompts import FewShotPromptTemplate

prompt = FewShotPromptTemplate(
    examples=examples,                      # example
    example_prompt=example_prompt,          # example formatting templates
    suffix="질문: {user_question}",         # suffix
    input_variables=["user_question"],      
)

prompt
```

* 실행 결과

```
FewShotPromptTemplate(input_variables=['user_question'], input_types={}, partial_variables={}, examples=[{'question': '2026년 1월 1일로부터 10일이 지난 날은?', 'answer': '20260111'}, {'question': '2026년 크리스마스는 언제일까요?', 'answer': '20261225'}, {'question': '내가 LangChain의 매력에 푹 빠지기 시작한 날은?', 'answer': '20260215'}], example_prompt=PromptTemplate(input_variables=['answer', 'question'], input_types={}, partial_variables={}, template='질문/답변: {question}\n{answer}'), suffix='질문: {user_question}')
```

## 3. Partial Prompt

**Partial Prompt** 는 **프롬프트 템플릿을 부분적으로 (partial) 포맷팅** 하는 것을 말한다.

* 즉, **필요한 값의 일부를 미리 입력하는 방식** 을 통해 새로운 프롬프트 템플릿을 만드는 것이다.

**1. 문자열 값을 이용한 partial formatting**

* 예시 코드

```python
template_text = "{topic}의 {discussion_topic}에 대해 {how} 알려줘."
prompt = PromptTemplate.from_template(template_text)
print('\noriginal prompt :\n', prompt)

# topic 만 채운 partial prompt
partial_prompt_1 = prompt.partial(topic="거대 언어 모델")
partial_prompt_1_formatted = partial_prompt_1.format(discussion_topic="언어 해석 역량", how="친절하게")
print('\npartial (1) :\n', partial_prompt_1)
print('\npartial (1) - formatted :\n', partial_prompt_1_formatted)

# discussion topic 까지 채운 partial prompt
partial_prompt_2 = partial_prompt_1.partial(discussion_topic="윤리적 문제")
partial_prompt_2_formatted = partial_prompt_2.format(how="간단히")
print('\npartial (2) :\n', partial_prompt_2)
print('\npartial (2) - formatted :\n', partial_prompt_2_formatted)
```

* 실행 결과

```
original prompt :
 input_variables=['discussion_topic', 'how', 'topic'] input_types={} partial_variables={} template='{topic}의 {discussion_topic}에 대해 {how} 알려줘.'

partial (1) :
 input_variables=['discussion_topic', 'how'] input_types={} partial_variables={'topic': '거대 언어 모델'} template='{topic}의 {discussion_topic}에 대해 {how} 알려줘.'

partial (1) - formatted :
 거대 언어 모델의 언어 해석 역량에 대해 친절하게 알려줘.

partial (2) :
 input_variables=['how'] input_types={} partial_variables={'topic': '거대 언어 모델', 'discussion_topic': '윤리적 문제'} template='{topic}의 {discussion_topic}에 대해 {how} 알려줘.'

partial (2) - formatted :
 거대 언어 모델의 윤리적 문제에 대해 간단히 알려줘.
```

**2. 문자열 값을 함수를 통해 반환하는 방식의 partial formatting**

* 예시 코드

```python
# 함수를 사용한 Partial Formatting

import random

def get_random_number():
    return random.randint(0, 9)

prompt = PromptTemplate(
    template="내가 선택한 숫자는 {random_number}, 이 숫자는 {meaning}을 뜻하지.",
    input_variables=["meaning"],
    partial_variables={"random_number": get_random_number}
)

print(prompt.format(meaning="행운"))
```

* 실행 결과

```
내가 선택한 숫자는 7, 이 숫자는 행운을 뜻하지.
```

## 4. 참고 링크

* [LLM 기초 - 프롬프트 엔지니어링](../AI%20Basics/LLM%20Basics/LLM_기초_Prompt_Engineering.md)
