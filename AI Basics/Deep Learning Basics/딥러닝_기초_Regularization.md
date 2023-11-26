## 목차
1. Reguarization 이란? 그 목적은?

2. L1, L2 Regularization

3. Gradient Vanishing

4. Batch Normalization

## Regularization 이란? 그 목적은?
딥러닝에서 **Regularization (정규화)** 란, 오버피팅 방지를 위한 일종의 규제를 의미한다.

오버피팅을 방지하기 위해서 validation dataset에서의 loss가 가장 낮은 (=좋은) 기록이 5회의 epoch 이상 갱신되지 않는 등 특정 조건을 만족시킬 때 **학습 조기 종료 (Early Stopping)** 를 할 수 있는데, 이것 역시 정규화의 일종이다.

## L1, L2 Regularization
L1, L2 Regularization은 **loss function의 값에 weight의 크기를 더해 주는 방식**의 정규화이다.

기존 loss function의 값이 $L_0$일 때, L1, L2 정규화에 의한 새로운 loss function의 수식은 다음과 같다. (w = weight)

* L1 Regularization
  * $L = L_0 + \displaystyle \frac{\lambda}{n} \sum_{w} |w|$

* L2 Regularization
  * $L = L_0 + \displaystyle \frac{\lambda}{n} \sum_{w} w^2$

이와 같은 정규화를 통해서 다음과 같은 효과를 얻을 수 있다.
* L1 정규화 : 손실함수를 미분하면 weight 부분은 실제 weight의 값에 관계없이 **절댓값이 일정하고 부호만 남아 있는 상수**가 된다. 가중치를 갱신할 때 이 값을 빼게 되므로, 신경망의 weight 값들 중 작은 값들은 0이 되고, **큰 (중요한) weight 들만 남게** 된다.
  * 따라서, 의미 있는 몇 개의 값이 중요한 역할을 하는 경우에 L1을 적용하면 좋다.
* L2 정규화 : 가중치 갱신 시 weight 값이 클수록 손실함수의 값에 이에 비례하는 큰 영향을 미치므로, **weight의 절댓값이 커지는 것 자체를 방지**할 수 있다.
  * 따라서 특정 가중치가 매우 커지는 것을 방지할 수 있다. 

## Gradient Vanishing
**Gradient Vanishing** 이란, 딥러닝 모델의 역전파 과정에서 입력층에 가까워질수록 gradient의 값이 0에 가까워져서 학습이 잘 되지 않는 현상을 말한다.

Gradient Vanishing의 원인은 일반적으로 **활성화 함수** 때문이다. 활성화 함수의 미분값이 1 미만인 경우, 역전파 과정에서 이 값이 계속 곱해지면서 입력층에 가까워질 때는 gradient가 0에 가까워지는 것이다.
* 딥러닝에서 많이 쓰이는 sigmoid 함수의 경우, 미분값의 최댓값은 x=0에서 0.25에 불과하다.
* tanh 함수는 미분값이 x=0에서 1로 최대이지만, x가 0이 아닐 때는 미분값으로 1보다 훨씬 작은 값을 갖는다.

Gradient Vanishing의 해결 방법은 다음과 같다.
* 특정 범위에서 미분값이 항상 1인 활성화 함수인 **ReLU = max(0, x)** 또는 **Leaky ReLU = max(0.05x, x)** 등을 사용한다.
* Batch Normalization을 사용한다.

## Batch Normalization
**Batch Normalization** 이란, 전체 데이터셋이 아닌 그 데이터셋을 나눈 작은 학습 단위인 Batch에 대해, **신경망 자체적으로** 그 평균과 분산을 이용하여 정규화하는 것이다.
* 각 레이어마다 Normalization Layer를 따로 두어서 분포가 변형되지 않게 한다.

배치 정규화를 통해 다음과 같은 효과를 얻을 수 있다.
* **Gradient vanishing 등을 방지**하여 안정적으로 학습할 수 있게 한다.
* 평균과 분산이 각 batch마다 달라지기 때문에 특정 weight가 매우 커지는 것을 방지할 수 있다.
