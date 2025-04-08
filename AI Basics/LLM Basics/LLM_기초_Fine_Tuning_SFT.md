## 목차

* [1. SFT (Supervised Fine-tuning)](#1-sft-supervised-fine-tuning)
  * [1-1. 예시](#1-1-예시) 
* [2. SFT 의 프로세스](#2-sft-의-프로세스)
* [3. SFT 를 위한 데이터셋 선택](#3-sft-를-위한-데이터셋-선택)
* [4. SFT 의 특징: Overfitting 이 오히려 좋다?](#4-sft-의-특징-overfitting-이-오히려-좋다)
  * [4-1. Overfitting 이 오히려 좋은 이유 (추정)](#4-1-overfitting-이-오히려-좋은-이유-추정)

## 1. SFT (Supervised Fine-tuning)

**SFT (Supervised Fine-tuning)** 은 거대 언어 모델에 대한 [Fine Tuning](LLM_기초_Fine_Tuning.md) 방법 중 하나로, **지도 학습을 이용한 Fine-tuning** 을 의미한다.

* 즉, 모델의 입력에 대해 특정한 정답이 제공되고, 이 정답과의 Loss 를 최소화하는 것이 목표이다.
* 학습을 위한 Loss Function 은 **생성된 문장 간 비교** 해야 하므로, Next token prediction 과 관련된 조건부 확률의 곱 또는 로그 합을 이용하는 방법 등을 사용할 수 있다.
  * 자세한 것은 [해당 문서](LLM_기초_Fine_Tuning.md#5-llm-fine-tuning-의-loss-function) 참고.

SFT 의 목적은 다음과 같다.

* 특정 domain 에 최적화된 데이터셋을 이용하여, LLM을 해당 domain 에 대한 사용자의 질문에 적절히 응답할 수 있도록 학습시킨다.
* 주어진 domain 에 알맞은 문장 형식이나 말투 등을 LLM 에게 학습시켜서, LLM 이 답변할 때 의도한 문장 형식이나 말투로 답변하게끔 한다.

SFT 의 장단점은 다음과 같다.

* 장점
  * 모델을 domain 에 최적화시켜서 **domain 에 대한 성능을 극강** 으로 올리기에 적합한 방법론임
  * 상황에 맞게 다양한 비즈니스적 요구를 만족시킬 수 있음
* 단점
  * 품질이 높은 labeled data 필요
  * 거대 언어 모델에 대한 SFT 의 경우 학습에 많은 자원이 소비될 수 있음

### 1-1. 예시

* **LLM 의 MBTI를 'T'에서 'F'로 바꾸기**

거대 언어 모델의 답변이 주어진 질문에 대해 해결 방법 위주로 답변할 때, 그 대신 공감하는 내용 위주로 답변할 수 있도록 지도 학습시킬 수 있다.

예를 들어, 기존 LLM 에서의 사용자 질문 (프롬프트) 및 이에 대한 답변이 다음과 같다고 하자.

| 사용자 질문                       | LLM 답변                                   |
|------------------------------|------------------------------------------|
| ```나 오늘 수학 시험에서 50점을 맞았어.``` | ```앞으로 예습 복습 까먹지 말고, 문제집 많이 푸는 게 중요해.``` |

이 모델에 다음과 같은 구성의 데이터셋으로 지도학습을 시키면,

| 입력 (사용자 질문)                        | 출력 (LLM 답변)                               |
|------------------------------------|-------------------------------------------|
| ```열쇠를 잃어버려서 집에 못 들어가고 있어.```      | ```아이고 저런... 열쇠 빨리 누가 찾아 줘야 할 텐데... ㅠㅠ``` |
| ```딥러닝 공부하는데 활성화 함수가 이해가 잘 안 돼.``` | ```그래도 하다 보면 언젠가는 잘 이해될 거야. 파이팅!```       |
| ...                                | ...                                       |

이후 Fine-tuning 된 LLM 에 다시 질문하면 아래와 같이 다른 답변을 얻을 수 있을 것이다.

| 사용자 질문                       | LLM 답변                                      |
|------------------------------|---------------------------------------------|
| ```나 오늘 수학 시험에서 50점을 맞았어.``` | ```열심히 노력했는데 어쩌다 ㅠㅠ 다음에는 잘 볼 거야! 걱정하지 마!``` |

## 2. SFT 의 프로세스

Supervised Fine-tuning 의 프로세스는 일반적으로 다음과 같다.

* 데이터셋 준비
  * 데이터를 수집하거나, 이미 수집된 품질 좋은 데이터를 사용할 수 있다.
  * 데이터셋 선택 가이드는 [이 부분](#3-sft-를-위한-데이터셋-선택) 참고.
* 모델 [하이퍼파라미터](../Machine%20Learning%20Models/머신러닝_방법론_HyperParam_Opt.md) 결정
* **모델 Fine-tuning 실시**
* 모델 테스트 및 평가 (필요 시 적절한 방법으로 성능 개선)
* 최종 모델 배포

## 3. SFT 를 위한 데이터셋 선택

Supervised Fine-tuning 을 위한 데이터셋을 선택할 때 다음을 고려해야 한다.

* 데이터의 양
  * 최소한 학습 자체가 되지 않을 정도로 양이 적지는 않아야 한다.
  * 단, **데이터의 양이 많다고 무조건 학습이 잘 되는 것은 아니다.**
* 데이터 형식
  * **다양할수록** 좋음
  * 이는 LLM 은 자체적으로 지식 및 답변 생성 능력을 가지고 있으므로, **다양한 형식의 데이터를 통해 LLM이 답변해야 할 적절한 형식을 학습시키는 것** 이 LLM Fine-tuning 의 핵심이기 때문이다.
* 데이터의 품질
  * **높을수록** 좋음
  * 저품질의 데이터는 LLM Fine-tuning 에 **오히려 악영향** 이 있을 수 있다.
  * 고품질의 데이터일수록 데이터의 양, 모델의 규모가 더 작아도 원하는 성능을 얻을 가능성이 높아진다.

## 4. SFT 의 특징: Overfitting 이 오히려 좋다?

SFT 의 특징 중 하나로 **Overfitting 이 발생하는 경우 사람이 느끼는 성능은 오히려 좋을 수 있다** 는 것이 있다. 이는 [해당 논문](https://arxiv.org/pdf/2203.02155) 의 **3.5 Models** 에 있는 다음 내용에 근거한다.

```text
Supervised fine-tuning (SFT).
We fine-tune GPT-3 on our labeler demonstrations using supervised learning.
We trained for 16 epochs, using a cosine learning rate decay, and residual
dropout of 0.2. We do our final SFT model selection based on the RM score on
the validation set. Similarly to Wu et al. (2021), we find that our SFT models
overfit on validation loss after 1 epoch; however, we find that training for
more epochs helps both the RM score and human preference ratings, despite this
overfitting.
```

[(출처)](https://arxiv.org/pdf/2203.02155) : Long Ouyang and Jeff Wu et al., Training language models to follow instructions with human feedback, 2022

### 4-1. Overfitting 이 오히려 좋은 이유 (추정)

여기서는 지도학습 되는 LLM 이 Next token prediction 관련 Loss Function 을 사용한다고 가정한다.

* 추정: **유의어 예측을 오답으로 간주**
  * LLM 이 next token prediction 에서 정답 단어가 아닌, **그 단어의 유의어를 예측했을 때 이를 오답으로 간주하여 Valid data의 Loss Function 의 값이 커지고,** 이를 overfitting 으로 간주한다는 의미이다.
  * 즉, LLM 이 overfitting 이 되도록 오랫동안 학습하면, **Training data 에서는 거의 정답을 하지만, Valid data 에서는 정답과 가까운 오답 단어를 예측** 한다는 뜻이다.
  * 그러나, 사람이 평가하기에는 **오답 단어도 또 다른 정답으로 느껴지기에**, 사람이 보기에는 LLM의 성능이 overfitting 되지 않은 모델보다 오히려 좋아지는 것이다. 

* 학습 데이터의 경우
  * overfitting 이 되기 전에는 아래 괄호 안에 들어갈 말을 ```model``` 로 예측하기도 했으나, overfitting 이 될 정도로 학습을 반복한 후에는 ```algorithm``` 이라고만 예측한다.

```
(정답) I think the best machine learning ( algorithm ) ...
================================================
(오답) I think the best machine learning ( model ) ...
```

* valid 데이터의 경우
  * overfitting 이 된 후, 아래 괄호에 들어갈 말을 ```statistics``` 가 아닌 ```math```, ```mathematics``` 와 같이 예측하기도 한다.
  * 그러나, 사람의 입장에서 보면 ```math```, ```mathematics``` 역시 정답일 가능성이 높다.

```
(정답) What we need to study to understand Deep Learning is ( statistics ) .
================================================
(오답1) What we need to study to understand Deep Learning is ( math ) .
(오답2) What we need to study to understand Deep Learning is ( mathematics ) .
```