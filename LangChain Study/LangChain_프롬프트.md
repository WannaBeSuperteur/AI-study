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
  * TBU 

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

### 2-2. ChatPromptTemplate

### 2-3. FewShotPromptTemplate

## 3. Partial Prompt

## 4. 참고 링크

* [LLM 기초 - 프롬프트 엔지니어링](../AI%20Basics/LLM%20Basics/LLM_기초_Prompt_Engineering.md)
