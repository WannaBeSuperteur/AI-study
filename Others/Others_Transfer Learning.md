# Transfer Learning (전이학습)

2024.01.29 작성중

## Transfer Learning 이란?
**Transfer Learning (전이학습)** 이란, 하나의 task에서 훈련된 모델을 다른 task를 훈련하기 위한 모델의 초기 모델로 사용하는 것을 의미한다.

즉, 첫 번째 task에서 학습된 모델을 통해서 다른 task를 모델링하는 경우 첫 번째 task에서 훈련한 내용 (예: 갱신된 신경망 가중치 등) 이 도움이 될 것으로 기대하여 이를 이용하는 것이다.

전이 학습의 특징은 다음과 같다.
* 기존보다 적은 양의 데이터로 높은 성능에 이르도록 모델을 훈련시킬 수 있다.
* Vision, NLP 등의 분야에서는 일반적인 내용을 학습한 모델을 사전 훈련된 모델로 사용할 수도 있다.

## 기존의 전통적인 머신러닝과의 차이
|기존 Machine Learning|Transfer Learning|
|---|---|
|계산 비용이 **많이** 소비됨|계산 비용이 **적음**|
|데이터가 **많이** 필요함|데이터가 **적게** 필요함|
|독립적으로 새로운 모델을 훈련|사전 훈련된 모델의 지식을 이용|

## 사전 학습 (Pre-training), 미세 조정 (Fine-tuning)
