# Zero-shot, One-shot, Few-shot Learning

2024.01.30 작성중

## X-shot Learning 이란? (X는 Zero, One, Few)
**X-shot Learning** 이란, **분류를 위한 머신러닝 모델** 에서 적은 양의 데이터만으로도 새로운 클래스를 예측하는 기술을 의미한다. 예를 들어, 개와 고양이를 분류하는, Convolutional Neural Network 기반 딥러닝 모델이 새로운 클래스인 토끼를 인식해야 하는데, 토끼에 대한 데이터가 부족하다면 이 기술을 활용하는 것이 적절하다.

X-shot Learning model에서의 X는 새로운 분류를 인식하기 위해 학습하는 데이터의 양을 의미하며, **Zero (0개), One (1개), Few (일반적으로 10개 이하)** 중 하나이다. 즉, 일반적인 분류 모델에서 수백 개 이상의 데이터를 사용하는 대신, 그보다 훨씬 적은 양의 데이터로도 학습이 되게 하는 기술이다.

## Zero-shot Learning
**Zero-shot Learning** 이란, **관련 데이터가 없는,** 즉 학습 데이터에 포함되어 있지 않은 (unseen) 클래스를 예측하는 기술이라고 할 수 있다. 머신러닝 모델이 unseen 데이터의 class를 예측하게 하려면 **semantic information** 이라는 것을 이용해야 한다.

**semantic information** 은 다음과 같이 구현할 수 있다.
* data의 class에 대한 word embedding
  * BERT 등 기존 NLP 모델로부터 사전에 학습된 word vector (예: '개'의 word vector, '고양이'의 word vector) 를 가져오는 방법을 사용할 수 있다.
* class의 특징을 이용
  * 예를 들어 '개'라는 class의 특징과 '고양이'라는 class의 특징을 각각 벡터화하면, 마찬가지로 unseen class인 '토끼'의 특징도 마찬가지로 벡터화할 수 있다. 이는 data class의 word embedding과 유사하다고 할 수 있다.

## One-shot Learning
**One-shot Learning** 이란, **오직 1개의 해당 class의 데이터** 만으로 새로운 class를 인식하는 기술을 말한다. 예를 들어 개와 고양이를 분류하는 모델에 토끼 사진 1장을 추가하여 '토끼'라는 새로운 class를 인식하게 하는 것을 말한다.

토끼 사진 1장을 '토끼'라는 class로 단순히 분류하여 모델을 그냥 학습시키면 **심각한 overfitting** 이 발생하기 때문에, 이와는 다른 방법을 사용해야 한다. One-shot Learning을 구현하기 위한 구체적인 기술은 다음과 같다.
* Siamese Network
* Triplet Loss

## Few-shot Learning
**Few-shot Learning** 은 One-shot Learning보다 다소 많은, 즉 일반적으로 **10개 이내의 해당 class의 데이터** 를 이용하여 새로운 class를 인식하는 기술이다.

기술적인 구현 방법은 One-shot Learning에서와 유사하게 Siamese Network, Triplet Loss 등을 이용할 수 있다.