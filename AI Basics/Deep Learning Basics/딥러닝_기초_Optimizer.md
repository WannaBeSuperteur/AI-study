## 목차
1. Optimizer 란?

2. Optimizer 의 종류
* 2-1. RMSProp
* 2-2. Adam
* 2-3. 기타

## Optimizer 란?
딥러닝에서 **최적화 (Optimization) 란, 손실 함수 (loss function) 를 줄여서 모델 예측의 오차를 줄이는 것**이다.

그렇다면 **Optimizer 란, 이 최적화를 수행하는 알고리즘**을 말한다.

## Optimizer 의 종류
Adam Optimizer, SGD, RmsProp 등이 잘 알려진 Optimizer 이다.

### RMSProp
기존 Gradient Descent 방법에서는 다음과 같은 방식으로 weight과 bias를 갱신했다.
* (weight) = (weight) - (learning rate) * (weight의 gradient)
* (bias) = (bias) - (learning rate) * (bias의 gradient)

RMSProp은 다음과 같은 방식으로 weight과 bias를 갱신한다.
* $${S_{dW}} = \beta * S_{dW} + (1 - \beta)dW^2$$
* $${S_{db}} = \beta * S_{db} + (1 - \beta)db^2$$
* $$(weight) = (weight) - (LearningRate) * \frac{(weight의 gradient)}{\sqrt{S_{dW} + \epsilon}}$$
* $$(bias) = (bias) - (LearningRate) * \frac{(bias의 gradient)}{\sqrt{S_{db} + \epsilon}}$$

이때 $S_{dW}$, $S_{db}$는 처음에 모두 0으로 초기화하고, $\beta$의 값은 0.9 등 1에 가까운 값을 사용한다. 보다 자세한 설명은 다음과 같다.

* $\beta$ : 지난 몇 회의 기울기 계산의 평균치를 사용할 것인지에 대한 가중치로, 그 횟수는 $\displaystyle \frac {1}{1 - \beta}$ 이다.
* $\epsilon$ : 계산식의 분모가 0이 되지 않게 하기 위한 작은 양수 값으로, 1억 분의 1, 100억 분의 1 등의 값을 사용한다.

### Adam
Adam은 $\alpha$, $\beta_1$, $\beta_2$, $\epsilon$ 이라는 4개의 하이퍼파라미터를 이용한다.

먼저 $v_{dW} = 0, S_{dW} = 0, v_{db} = 0, S_{db} = 0$으로 초기화한 후, 다음과 같은 계산식을 통해 최적화를 진행한다.
* $v_{dW} = \beta_1 v_{dW} + (1 - \beta_1)dW$
* $v_{db} = \beta_1 v_{db} + (1 - \beta_1)db$
* $S_{dW} = \beta_2 S_{dW} + (1 - \beta_2)dW^2$
* $S_{db} = \beta_2 S_{db} + (1 - \beta_2)db^2$

RMSProp과 달리, 다음과 같이 bias correction을 적용한다.
* $v_{dW}^{bc} = \displaystyle \frac {v_{dW}}{1 - \beta^t}$
* $v_{db}^{bc} = \displaystyle \frac {v_{db}}{1 - \beta^t}$
* $S_{dW}^{bc} = \displaystyle \frac {S_{dW}}{1 - \beta^t}$
* $S_{db}^{bc} = \displaystyle \frac {S_{db}}{1 - \beta^t}$

마지막으로 RMSProp의 가중치 갱신 방식을 유사하게 적용한다. (단, gradient 대신 bias correction 된 값을 적용한다.)
* $$(weight) = (weight) - (LearningRate) * \frac{v_{dW}^{bc}}{\sqrt{S_{dW}^{bc} + \epsilon}}$$
* $$(bias) = (bias) - (LearningRate) * \frac{v_{db}^{bc}}{\sqrt{S_{db}^{bc} + \epsilon}}$$

여기서 $\beta_1, \beta_2$는 각각 1차, 2차 moment이며, $\epsilon$은 RMSProp과 동일한 목적으로 사용한다.

### 기타
이 외의 Optimizer로 다음과 같은 것들이 있다.
* AdaGrad, AdaDelta, Nadam, Adabelief
* GD (Gradient Descent), SGD (Stochastic Gradient Descent), Batch Gradient Descent
* Momentum
* NAG (Nesterov Accelerated Gradient)