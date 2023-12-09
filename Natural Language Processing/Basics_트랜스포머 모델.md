# 자연어 처리 (NLP) 기초 - 트랜스포머 (Transformer) 모델

2023.12.09 작성중

## 트랜스포머 모델
**트랜스포머 (Transformer)** 는 인코더에서 단어 시퀀스를 입력받고, 디코더에서 출력 단어 시퀀스를 출력하는 형태의 Encoder-Decoder 구조를 유지하면서, **N개의 Encoder와 N개의 Decoder** 를 이용하는 형태의 모델이다.

![트랜스포머 모델 기본 구조](./images/Transformer_1.PNG)

## 포지셔널 인코딩 (Positional Encoding)
RNN에서는 단어를 순서대로 입력받았기 때문에 자연어 처리에 유용했으나, 트랜스포머에서는 그렇지 않기 때문에 **단어의 순서를 따로 표시** 해야 한다. 그 방법이 **포지셔널 인코딩 (Positional Encoding)** 이다.

트랜스포머에서 입력 단어 시퀀스 내에서의 특정 임베딩 벡터의 위치를 $p$, 임베딩 벡터 내에서 특정 차원의 인덱스를 $i$, 임베딩 벡터의 차원을 $d_model$ 이라고 하자. 이때 $i$의 값이 짝수 $2i$인지, 홀수 $2i+1$인지에 따라 포지셔널 인코딩 값 PE가 각각 다음과 같다.

$PE_{(p, 2i)} = sin((pos/(10000^{2i/{d_model}}))$
$PE_{(p, 2i+1)} = cos((pos/(10000^{2i/{d_model}}))$

예를 들어 입력 단어 시퀀스가 20개의 토큰으로 구성되어 있고 ($p=0,1,...,19$), 임베딩 벡터가 32차원일 때 ($d_model=32, i=0,1,...,31$), $i$와 $p$의 값에 따른 Positional Encoding 값은 다음과 같다.

![트랜스포머 포지셔널 인코딩 값](./images/Transformer_2.PNG)

## GPT (Generative Pre-trained Transformer)
**GPT (Generative Pre-trained Transformer)** 는 그대로 풀어 쓰면 **사전 학습된 생성형 트랜스포머** 이다.
* 생성 (Generative) 은 생성형 AI 기술을 적용하여 답변을 생성한다는 의미이다.
* 사전 학습 (Pre-trained) 는 사전 학습된 언어 모델을 사용한다는 의미이다.

GPT의 역사는 다음과 같다.

2022년 11월 30일, GPT의 응용으로 전 세계 AI의 역사에 한 획을 그은 **ChatGPT** 가 등장했으며, 2023년 3월 GPT-4가 적용되었다.