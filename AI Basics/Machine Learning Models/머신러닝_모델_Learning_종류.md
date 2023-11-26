## 목차
1. 머신러닝 모델의 학습 방식에 따른 분류

2. Supervised Learning (지도학습)

3. Unsupervised Learning (비지도학습)

4. Semi-supervised Learning (반지도학습)

5. Reinforcement Learning (강화학습)

## 머신러닝 모델의 학습 방식에 따른 분류
머신러닝 모델은 그 학습 방식에 따라 다음과 같이 분류한다.
* Supervised Learning (지도학습)
* Unsupervised Learning (비지도학습)
* Semi-supervised Learning (반지도학습)
* Reinforcement Learning (강화학습)

## Supervised Learning (지도학습)
**Supervised Learning (지도학습)** 이란, **입력값과 출력값이 주어진** 학습 데이터를 이용해서 학습한 후, 새로운 입력값이 있는 데이터에 대해 그 출력값을 예측하는 형태의 머신러닝 모델이다.

지도학습 모델을 통해서 해결할 수 있는 문제는 크게 **회귀, 분류** 의 2가지로 나눌 수 있다.

* **Regression (회귀) 문제** : 출력값이 연속된 숫자 형태일 때, 새로운 입력 데이터에 대해 그 오차를 최소화하도록 모델을 학습한다.
  * 예시: 고객 및 경제 상황 등의 데이터에 기반한 판매량 예측, 최근 3개월 간의 날짜별 기온이 주어졌을 때 다음 날의 기온 예측

* **Classification (분류) 문제** : 출력값이 분류 형태일 때, 새로운 입력 데이터의 분류를 예측한다.
  * 딥 러닝을 비롯한 머신러닝 모델에서 분류를 출력 데이터로 할 때는, 보통 **one-hot encoding** (해당 class와 매칭되는 index의 값은 1, 나머지 index의 값은 모두 0) 을 이용하여 출력 데이터를 만든다.
    * 이때, 모델의 예측 출력값은 입력 데이터가 해당 class일 **확률**이라고 할 수 있다.
  * 예시: 개와 고양이 등 동물 분류, 붓꽃의 setosa, versicolor, virginica 분류, 자연어 처리 (NLP) 에서의 감정 분석

지도학습 모델의 예시는 다음과 같다.
* K-Nearest Neighbor
* Decision Tree
* Naive Bayes 모델
* 입력층과 출력층이 있는 형태의 일반적인 딥러닝 모델

## Unsupervised Learning (비지도학습)
**Unsupervised Learning (비지도학습)** 이란, **입력값과 출력값의 구분이 따로 없는 데이터** 를 학습 후 군집화한 다음, 새로운 데이터에 대해 그 데이터가 속하는 군집(cluster)을 예측하는 모델이다.

비지도학습 모델의 예시는 다음과 같다.
* K-means Clustering
* DBSCAN
* PCA (Principal Component Analysis)
* ChatGPT, 그림 그리는 AI 등 **생성형 AI** 에 사용되는 비지도학습 모델
  * GAN (Generative Adversarial Network)
  * AutoEncoder
  * VAE (Variational AutoEncoder)

## Semi-supervised Learning (반지도학습)
**Seml-supervised Learning (반지도학습)** 이란, **입력값은 있지만 출력값은 없는 데이터를 지도학습에 사용**하는 것이다. 이때 다음과 같이 가정한다.
* 출력값이 없는 데이터들 중 입력값이 서로 유사한 데이터들은 출력값도 유사할 것이다.
  * 입력값이 서로 유사하다는 것은, K-means Clustering 등 비지도학습 알고리즘 사용 결과 같은 군집에 속하는 데이터가 이에 해당한다.

## Reinforcement Learning (강화학습)
**Reinforcement Learning (강화학습)** 이란, **환경 (environment)** 안에 있는 **AI 모델 (agent)** 이, **현재 상태 (state)** 에서의 **행동 (action)** 에 대한 **보상 (reward)** 을 최적화하기 위해, 수많은 행동을 반복하면서 최적의 행동 방향성을 찾는 학습 알고리즘이다.
* 사람이 학습하는 방식인 '시행착오' 방법과 매우 유사하다.
* 제한된 데이터를 학습하는 것이 아니라 시행착오를 통해 AI 모델이 스스로 데이터를 생성(?)하면서 학습한다. 따라서 학습 데이터가 부족할 때에도 환경 (environment) 및 보상 (reward) 을 적절히 설정하기만 하면 된다.

강화학습을 구현하는 알고리즘은 다음과 같다.
* Q-learning
  * Deep Q-learning (Q-learning에 딥러닝을 적용한 것)

강화학습을 사용하는 예시는 다음과 같다.
* 온라인 게임의 NPC 등 각종 게임
  * 딥마인드의 알파고 (AlphaGo), 알파 제로 (Alpha Zero) 등
* 로보틱스
* 자율주행 자동차