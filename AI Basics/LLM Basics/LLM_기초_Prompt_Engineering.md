## 목차

* [1. User Prompt](#1-user-prompt)
* [2. Prompt Engineering](#2-prompt-engineering)
* [3. Prompt Engineering 방법의 종류](#3-prompt-engineering-방법의-종류)
  * [3-1. Zero-Shot Prompting](#3-1-zero-shot-prompting)
  * [3-2. Few-Shot Prompting](#3-2-few-shot-prompting)
  * [3-3. Directional Stimulus Prompting](#3-3-directional-stimulus-prompting)
  * [3-4. RAG (Retrieval Augmented Generation)](#3-4-rag-retrieval-augmented-generation)
  * [3-5. Chain-of-Thought Prompting](#3-5-chain-of-thought-prompting)
* [4. Prompt Engineering 의 한계점](#4-prompt-engineering-의-한계점)

## 1. User Prompt

**사용자 프롬프트 (User Prompt)** 는 거대 언어 모델의 사용자 입력으로 들어가는 자연어 데이터를 의미한다.

## 2. Prompt Engineering

**프롬프트 엔지니어링 (Prompt Engineering)** 이란, LLM 이 사용자가 원하는 답변을 출력하게끔 **사용자 프롬프트를 설계** 하는 것이다.

* LLM에 대한 일종의 '사용법'으로 볼 수 있다.

프롬프트 엔지니어링의 장점은 다음과 같다.

* AI에 대한 전문 지식 없이도 가능하다.
* 그러면서도 사용자가 의도한 형태의 답을 LLM이 할 수 있도록 한다.
* LLM의 답변을 **파싱 (Parsing) 알고리즘을 통해 해석** 해야 하는 경우 이를 위한 통일된 포맷으로 답변하게 할 수 있다.
  * 매번 LLM을 생성할 때마다 답변의 포맷이 달라져서, Parsing 이 불가능하여 오류가 발생하는 것을 방지

## 3. Prompt Engineering 방법의 종류

널리 알려진 Prompt Engineering 의 종류에는 다음과 같은 것들이 있다.

| 방법론                                  | 설명                                 |
|--------------------------------------|------------------------------------|
| Zero-Shot Prompting                  | 단순히 프롬프트만 입력                       |
| Few-Shot Prompting                   | 몇 가지 예시를 통해 LLM 답변의 정확도 향상         |
| Directional Stimulus Prompting       | LLM에게 문제 해결 방법에 대한 힌트 제공           |
| RAG (Retrieval Augmented Generation) | **문서를 바탕으로** LLM이 답변하게 함           |
| Chain-of-Thought Prompting           | **단계적 사고를 바탕으로** LLM이 추론적 답변을 하게 함 |

### 3-1. Zero-Shot Prompting

**Zero-Shot Prompting** 은 LLM에 단순히 User Prompt 만 입력하는 것이다.

### 3-2. Few-Shot Prompting

**Few-Shot Prompting** 은 LLM이 답변을 제공할 때 참고할 수 있도록 **몇 가지 예를 드는** 방법이다.

* 예시

```text
입력 node가 2개, hidden layer 2개의 node가 각각 4개, 6개, output layer 의 node가 1개인 딥러닝 모델을 node를 다음과 같은 형식으로 나타내 줘.
이때 각 node 를 "[고유 번호, 모양, 연결선 모양, 색상, 연결선이 가리키는 다른 node의 고유 번호들]" 의 형식으로 나타낸다.

예를 들어, "[1, 직사각형, 실선 화살표, #008000, [2, 3]]" 과 같이 나타낼 수 있다.
```

* [실제 ChatGPT 답변 링크](https://chatgpt.com/share/67ca8616-60a0-8010-b786-2dceb8073dd2)

### 3-3. Directional Stimulus Prompting

**Directional Stimulus Prompting** 은 LLM에게 **힌트를 제공** 하여, 사용자가 원하는 답변을 하도록 유도한다.

* 예시

```text
입력 node가 2개, hidden layer 2개의 node가 각각 4개, 6개, output layer 의 node가 1개인 딥러닝 모델을 node를 다음과 같은 형식으로 나타내 줘.
이때 각 node 를 "[고유 번호, 모양, 연결선 모양, 색상, 연결선이 가리키는 다른 node의 고유 번호들]" 의 형식으로 나타낸다.

이때, 색상은 RGB 색상 코드 형식으로 나타낸다.
```

* [실제 ChatGPT 답변 링크](https://chatgpt.com/share/67ca8694-2534-8010-b1fc-ac6da170c142)

### 3-4. RAG (Retrieval Augmented Generation)

**RAG (Retrieval Augmented Generation)** 은 **LLM 이 보유한 지식의 외부에 있는 문서에서 정보를 찾아서 (Retrieval)** 그 정보를 바탕으로 답변을 하는 것을 말한다.

프롬프트 엔지니어링에서는 보통 그 문서의 내용을 프롬프트에 포함하는 것을 의미한다.

* 자세한 것은 [해당 문서](LLM_기초_RAG.md) 참고.

### 3-5. Chain-of-Thought Prompting

**Chain-of-Thought (CoT) Prompting** 은 LLM 이 **단계적 사고를 바탕으로 추론적으로** 답변하게 하는 것을 말한다.

* CoT 에 대한 자세한 내용은 [해당 문서](LLM_기초_Chain_of_Thought.md) 참고.
* CoT 를 이용한 Prompt Engineering 에 대한 자세한 내용은 [해당 문서](LLM_기초_Chain_of_Thought.md#3-1-chain-of-thought-prompting) 참고.

## 4. Prompt Engineering 의 한계점

프롬프트 엔지니어링은 다음과 같은 한계점을 내포하고 있다.

* 공격자가 악의적인 프롬프트를 LLM에 전송하여, 제품 또는 서비스에 적용된 프롬프트 엔지니어링 방법론을 알고 탈취할 수 있다.
* OpenAI의 GPT 시리즈와 같이 API 호출 비용이 있는 경우, **프롬프트 엔지니어링으로 인해 추가된 토큰** 으로 인해 비용이 **더 비싸질** 수 있다.