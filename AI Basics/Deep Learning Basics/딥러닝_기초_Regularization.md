## 목차
* [1. Reguarization 이란? 그 목적은?](#1-regularization-이란-그-목적은)
* [2. L1, L2 Regularization](#2-l1-l2-regularization)
* [3. Gradient Vanishing](#3-gradient-vanishing)
* [4. Batch/Layer Normalization](#4-batchlayer-normalization)
  * [4-1. Batch Normalization](#4-1-batch-normalization) 
  * [4-2. Layer Normalization](#4-2-layer-normalization)

## 1. Regularization 이란? 그 목적은?

딥러닝에서 **Regularization (정규화)** 란, **[오버피팅](딥러닝_기초_Overfitting_Dropout.md) 방지를 위한 일종의 규제** 를 의미한다.

오버피팅을 방지하기 위해서 validation dataset에서의 loss가 가장 낮은 (=좋은) 기록이 5회의 epoch 이상 갱신되지 않는 등 특정 조건을 만족시킬 때 **학습 조기 종료 (Early Stopping)** 를 할 수 있는데, 이것 역시 정규화의 일종이다.

## 2. L1, L2 Regularization

L1, L2 Regularization은 **loss function의 값에 weight의 크기를 더해 주는 방식**의 정규화이다.

기존 loss function의 값이 $L_0$일 때, L1, L2 정규화에 의한 새로운 loss function의 수식은 다음과 같다. (w = weight)

* L1 Regularization
  * $L = L_0 + \displaystyle \frac{\lambda}{n} \sum_{w} |w|$

* L2 Regularization
  * $L = L_0 + \displaystyle \frac{\lambda}{n} \sum_{w} w^2$

![image](images/Regularization_1.PNG)

이와 같은 정규화를 통해서 다음과 같은 효과를 얻을 수 있다.

| 정규화    | 요약                 | 상세                                                                                                                                                                                                           |
|--------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| L1 정규화 | 중요한 weight 들만 유지   | **의미 있는 몇 개의 값이 중요한 역할을 하는 경우** 에 적용하면 좋음<br>- 손실함수를 미분하면 weight 부분은 실제 weight의 값에 관계없이 **절댓값이 일정하고 부호만 남아 있는 상수**가 됨<br>- 가중치를 갱신할 때 이 값을 빼게 되므로, 신경망의 weight 값들 중 작은 값들은 0이 되고, **큰 (중요한) weight 들만 남게** 됨 |
| L2 정규화 | 가중치 절대 크기 매우 커짐 방지 | **weight의 절댓값을 제어하여 overfitting 방지**<br>- 가중치 갱신 시 weight 값이 클수록 손실함수의 값에 이에 비례하는 큰 영향<br> - 따라서 특정 가중치가 매우 커지는 것을 방지 가능                                                                                     |

* L1, L2 정규화는 **weight 을 모델의 예측값으로 바꾸면**, [Loss Function](딥러닝_기초_Loss_function.md) 중 각각 **Mean Squared Error, Mean Absolute Error** 와 관련 있다.

## 3. Gradient Vanishing

**Gradient Vanishing** 이란, 딥러닝 모델의 역전파 과정에서 **입력층에 가까워질수록 gradient의 값이 0에 가까워져서 학습이 잘 되지 않는 현상** 을 말한다.

* Gradient 의 값이 매우 감소하기 때문에 학습 속도가 자연스럽게 느려진다.

Gradient Vanishing의 원인은 일반적으로 **[활성화 함수](딥러닝_기초_활성화_함수.md)** 때문이다. **활성화 함수의 미분값이 1 미만** 인 경우, 역전파 과정에서 이 값이 계속 곱해지면서 입력층에 가까워질 때는 gradient가 0에 가까워지는 것이다.
* 딥러닝에서 많이 쓰이는 sigmoid 함수의 경우, 미분값의 최댓값은 x=0에서 0.25에 불과하다.
* tanh 함수는 미분값이 x=0에서 1로 최대이지만, x가 0이 아닐 때는 미분값으로 1보다 훨씬 작은 값을 갖는다.

![image](images/Regularization_2.PNG)

Gradient Vanishing의 해결 방법은 다음과 같다.
* 특정 범위에서 미분값이 항상 1인 활성화 함수인 **ReLU = max(0, x)** 또는 **Leaky ReLU = max(0.05x, x)** 등을 사용한다.
* [Batch Normalization](#4-1-batch-normalization) 을 사용한다.

## 4. Batch/Layer Normalization

overfitting 을 방지하기 위한 대표적인 방법으로 **Normalization** 이 있다. Normalization 은 **Batch Normalization 과 Layer Normalization** 으로 구분된다.

* 둘 모두 **레이어 단위** 로 적용된다.

| 구분                  | 설명                                                                      |
|---------------------|-------------------------------------------------------------------------|
| Batch Normalization | 특정 레이어에 대해, 한 batch 내에서 **동일한 위치의 feature 의 값** 끼리 Normalization        |
| Layer Normalization | 특정 레이어에 대해, 각 sample 마다 **그 sample 내의 모든 feature 값** 에 대해 Normalization |

![image](images/Regularization_3.PNG)

### 4-1. Batch Normalization

**Batch Normalization** 이란, 전체 데이터셋이 아닌 그 데이터셋을 나눈 **작은 학습 단위인 Batch** 에 대해, **신경망 자체적으로** 그 평균과 분산을 이용하여 정규화하는 것이다.
* 각 레이어마다 **Normalization Layer** 를 따로 두어서 분포가 변형되지 않게 한다.

![image](images/Regularization_4.PNG)

배치 정규화를 통해 다음과 같은 효과를 얻을 수 있다.
* **Gradient vanishing 등을 방지**하여 안정적으로 학습할 수 있게 한다.
* 평균과 분산이 각 batch마다 달라지기 때문에 특정 weight가 매우 커지는 것을 방지할 수 있다.

### 4-2. Layer Normalization

**Layer Normalization** 은 Batch Normalization 과 달리 **같은 sample 의 모든 feature 에 대한 평균 및 표준편차** 를 이용하여 정규화하는 것을 이용한다.

* Batch Normalization 과 마찬가지로 Normalization Layer 를 따로 둔다.
* 역시 Gradient Vanishing을 방지할 수 있다.

Batch Normalization 과 비교한 Layer Normalization 의 장점은 다음과 같다.

* batch size가 작을 때도 안정적인 결과가 나온다.
* [Recurrent Neural Network (RNN)](../../Natural%20Language%20Processing/Basics_RNN과%20LSTM,%20GRU.md#rnn이란) 을 NLP 에 사용할 때, Normalization 대상 데이터 개수가 고정되어 있어서 더 효과적이다.