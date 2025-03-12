# 자연어 처리 (NLP) 기초 - 트랜스포머 (Transformer) 모델

## 목차

* [1. 트랜스포머 모델](#1-트랜스포머-모델)
* [2. 포지셔널 인코딩 (Positional Encoding)](#2-포지셔널-인코딩-positional-encoding)
* [3. 트랜스포머에서의 어텐션 메커니즘](#3-트랜스포머에서의-어텐션-메커니즘)
  * [3-1. Encoder Self-Attention](#3-1-encoder-self-attention)
  * [3-2. Masked Decoder Self-Attention](#3-2-masked-decoder-self-attention)
  * [3-3. Encoder-Decoder Attention](#3-3-encoder-decoder-attention)
* [4. Position-wise Feed Forward](#4-position-wise-feed-forward)
* [5. GPT (Generative Pre-trained Transformer)](#5-gpt-generative-pre-trained-transformer)
* [6. Vision 에의 응용 : Vision Transformer (ViT)](#6-vision-에의-응용--vision-transformer-vit)

## 1. 트랜스포머 모델

**트랜스포머 (Transformer)** 는 인코더에서 단어 시퀀스를 입력받고, 디코더에서 출력 단어 시퀀스를 출력하는 형태의 Encoder-Decoder 구조를 유지하면서, **N개의 Encoder와 M개의 Decoder** 를 이용하는 형태의 모델이다.

![트랜스포머 모델 기본 구조](./images/Transformer_1.PNG)

조금 더 깊이 들어가 보면 다음과 같은 구조를 가지고 있다.

* **입력, 출력** 문장의 각 token 을 **각각의 Encoder, Decoder** 로 매칭시킨다.
  * 각 입력/출력 token 에 Embedding + Positional Encoding 을 적용한다.
* 세부 동작 방식
  * 입력 문장은 Attention + Position-wise Feed Forward 를 거쳐 다음 Encoder 로 입력된다. 이것이 **입력 문장이 끝날 때까지 반복** 된다.
  * 마지막 Encoder 의 출력은 첫 번째 Decoder 의 Encoder-Decoder Attention 의 입력으로 들어간다.
  * Decoder 역시, N 번째 Decoder 의 출력이 N + 1 번째 Decoder 의 입력으로 들어가는 것이 **출력 문장이 끝날 때까지** 반복된다.

![image](images/Transformer_6.PNG)

## 2. 포지셔널 인코딩 (Positional Encoding)

RNN에서는 단어를 순서대로 입력받았기 때문에 자연어 처리에 유용했으나, 트랜스포머에서는 그렇지 않기 때문에 (즉, **모든 token 을 한번에 입력** 받기 때문에) **단어의 순서를 따로 표시** 해야 한다. 그 방법이 **포지셔널 인코딩 (Positional Encoding)** 이다.

트랜스포머에서 입력 단어 시퀀스 내에서의 특정 임베딩 벡터의 위치를 $p$, 임베딩 벡터 내에서 특정 차원의 인덱스를 $i$, 임베딩 벡터의 차원을 $d_model$ 이라고 하자. 이때 $i$의 값이 짝수 $2i$인지, 홀수 $2i+1$인지에 따라 포지셔널 인코딩 값 PE가 각각 다음과 같다.

* $PE_{(p, 2i)} = sin((pos/(10000^{2i/{d_{model}}}))$
* $PE_{(p, 2i+1)} = cos((pos/(10000^{2i/{d_{model}}}))$

예를 들어 입력 단어 시퀀스가 20개의 토큰으로 구성되어 있고 ($p=0,1,...,19$), 임베딩 벡터가 32차원일 때 ($d_{model}=32, i=0,1,...,31$), $i$와 $p$의 값에 따른 Positional Encoding 값은 다음과 같다.

![트랜스포머 포지셔널 인코딩 값](./images/Transformer_2.PNG)

Transformer 모델에서는 임베딩 벡터가 입력되기 전에 **Positional Encoding 값을 먼저 더한다.** 예를 들어 다음과 같다.

![Positional Encoding 예시](./images/Transformer_3.PNG)

즉, 단어가 Encoder 또는 Decoder의 hidden layer로 입력되기 전에 먼저 Positional Encoding 값이 더해진다.

## 3. 트랜스포머에서의 어텐션 메커니즘

트랜스포머 모델에서는 다음과 같은 3가지 [어텐션 메커니즘](Basics_어텐션%20(Attention).md) 이 사용된다.

| 어텐션 메커니즘                      | 위치      | 설명                                          |
|-------------------------------|---------|---------------------------------------------|
| Encoder Self-Attention        | Encoder | 입력 시퀀스 내에서 단어 간 관계 고려 (자기 자신을 포함한 모든 단어 고려) |
| Masked Decoder Self-Attention | Decoder | 출력 시퀀스에서 단어 간 관계 고려 (단, 매 순간 이전 단어만 고려)     |
| Encoder-Decoder Attention     | Decoder | 출력 시퀀스에서 입력 시퀀스의 단어를 고려한 Attention          |

각 어텐션을 그림으로 나타내면 다음과 같다.

![트랜스포머에서의 어텐션 1](./images/Transformer_4.PNG)

여기서 ```E1, E2, ..., En```은 각각 Encoder, ```D1, D2, ..., Dm```은 각각 Decoder를 나타낸다.

각 어텐션이 Encoder와 Decoder에서 실시되는 것을 그림으로 나타내면 다음과 같다.

![트랜스포머에서의 어텐션 2](./images/Transformer_5.PNG)

여기서 N 은 Encoder의 개수, M 은 Decoder의 개수를 나타낸다.

### 3-1. Encoder Self-Attention

**Encoder Self-Attention** 은 **입력 문장의 단어 (token) 간 관계** 를 고려한 Attention 이며, **Encoder** 에 위치한다.

* 이때 **Multi-head Attention** 를 적용하여, 입력 token 을 임베딩한 벡터의 차원을 **여러 개의 head 에 균등하게 분배** 한다.
* 단어 간 관계는 **자기 자신을 포함** 한다.

Self-Attention 의 효과는 다음과 같다.

* **입력 문장 내의 각 token 간 관계** 에 대한 분석
* 이를 통해, 지시대명사가 가리키는 것 등을 AI 모델이 파악할 수 있음

Encoder Self-Attention 을 수식으로 나타내면 다음과 같다.

* $\displaystyle SelfAttention(Q, K, V) = Softmax(\frac{Q \times K^T}{\sqrt {d_k}}) V$
  * $Q, K, V$ : 각각 Query, Key, Value
  * $d_k$ : 각 token 을 나타내는 Key 벡터의 차원 (Query, Value 벡터와 차원 동일)
* Encoder Self-Attention Matrix 의 shape 는 **(token 개수, $d_k$)** 이다.

----

Encoder Self-Attention 에 대해 보다 깊이 들어가면 다음과 같다.

![image](images/Transformer_7.PNG)

Attention Head 의 개수가 **8개** (예시) 이고, 문장의 입력 token 이 $N$ 개일 때, 다음과 같이 동작한다.

* 각 단어 벡터가 **512 차원** (예시) 인 Embedding + Pos Embedding 에 가중치 행렬을 곱해서 **$d_k$ = (512 / Head 개수) = 64 차원** 의 Query, Key, Value 행렬을 만든다.
* $N \times N$ 의 $QK^T$ 행렬을 구한다.
* $QK^T$ 행렬을 $d_k$ 의 제곱근으로 나누어 Attention Score 행렬 ($N \times N$) 을 구한다.
* $QK^T$ 행렬에 Value 에 해당하는 행렬 $V$ 를 곱하여 $N \times 64$ 의 Attention Matrix 를 구한다.
* 나머지 7개의 head 에서 계산된 Attention Matrix 들과 concatenate 한다.
* 마지막으로 가중치 행렬 $W_O$ 를 지나고, 이것이 Multi-head Self-Attention 의 최종 출력값이다.

### 3-2. Masked Decoder Self-Attention

### 3-3. Encoder-Decoder Attention

## 4. Position-wise Feed Forward

## 5. GPT (Generative Pre-trained Transformer)

**GPT (Generative Pre-trained Transformer)** 는 OpenAI의 Large Language Model (거대 언어 모델)로, 그대로 풀어 쓰면 **사전 학습된 생성형 트랜스포머** 이다.
* 생성 (Generative) 은 생성형 AI 기술을 적용하여 답변을 생성한다는 의미이다.
* 사전 학습 (Pre-trained) 는 사전 학습된 언어 모델을 사용한다는 의미이다.

GPT의 역사는 다음과 같다.
* OpenAI의 첫 LLM인 GPT-1이 2018년 6월 출시되었다.
* GPT-2가 2019년 2월 출시되었다. 단문의 경우 인간이 쓴 글과 구별할 수 없을 정도의 글을 쓸 수 있다.
* GPT-3가 2020년 6월 출시되었다.
* GPT-3.5가 2022년 11월 출시되었다.
* GPT-4가 2023년 3월 14일 출시되었다.

2022년 11월 30일, GPT의 응용으로 전 세계 AI의 역사에 한 획을 그은 **ChatGPT** 가 등장했으며, 2023년 3월 GPT-4가 적용되었다. (GPT-4 API는 2023년 7월부터 일반인이 사용 가능하다.)

## 6. Vision 에의 응용 : Vision Transformer (ViT)

Transformer 의 구조를 Vision 분야에 응용한 모델로 **Vision Transformer (ViT)** 가 있다. 자세한 것은 [해당 문서](../Image%20Processing/Basics_Vision_Transformer_ViT.md) 참고.