## 목차
* [1. Optimizer 란?](#1-optimizer-란)
* [2. Optimizer 의 종류](#2-optimizer-의-종류)
  * [2-1. RMSProp](#2-1-rmsprop)
  * [2-2. Adam](#2-2-adam)
  * [2-3. AdamW](#2-3-adamw)
  * [2-4. AdaGrad](#2-4-adagrad)
  * [2-5. AdaDelta](#2-5-adadelta)
  * [2-6. SGD (Stochastic Gradient Descent)](#2-6-sgd-stochastic-gradient-descent)
  * [2-7. 기타](#2-7-기타)
* [3. 탐구: 어떤 Optimizer 가 적절할까?](#3-탐구-어떤-optimizer-가-적절할까)
  * [3-1. 실험 설계](#3-1-실험-설계)
  * [3-2. 실험 결과](#3-2-실험-결과)

## 코드

## 1. Optimizer 란?
딥러닝에서 **최적화 (Optimization) 란, 손실 함수 (loss function) 를 줄여서 모델 예측의 오차를 줄이는 것**이다.

그렇다면 **Optimizer 란, 이 최적화를 수행하는 알고리즘**을 말한다.

## 2. Optimizer 의 종류
Adam Optimizer, SGD (Stochastic Gradient Descent), RmsProp 등이 잘 알려진 Optimizer 이다. 이들 잘 알려진 Optimizer 들에 대해 간단히 설명하면 다음과 같다.

| Optimizer | 핵심 아이디어                                                                                                                                                                       |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RMSProp   | - Gradient 가 큰 매개변수일수록 learning rate 낮음                                                                                                                                       |
| Adam      | - RMSProp **+ Momentum**<br>- Momentum : Gradient 갱신 방향에 '가속'을 붙이는 방식                                                                                                         |
| AdamW     | - Adam 에서 **weight decay 를 Loss Function 에서 분리**<br>- weight decay 가 적용된 [L2 Regularization](딥러닝_기초_Regularization#2-l1-l2-reguliarization) 을 Adam Optimizer가 처리할 때의 성능 저하 해결 |
| AdaGrad   | - 가끔 업데이트되는 파라미터일수록 learning rate 높음<br>- 이로 인해, 가끔 업데이트되는 파라미터가 많은 sparse dataset 에서 효과적                                                                                     |
| AdaDelta  | - **지수 이동 평균 (EMA)** 를 이용, **최근의 Gradient** 의 가중치를 더 높임<br>- 시간이 지남에 따라 가중치는 지수적으로 감소                                                                                         |
| SGD       | - 전체 데이터셋 대신 **1개 샘플 또는 batch** 를 학습하여 파라미터 업데이트<br>- 대규모 데이터셋에서 계산 비용 절감 효과                                                                                                  |

### 2-1. RMSProp

핵심 아이디어
* Gradient 가 큰 매개변수의 학습 속도 (learning rate) 를 줄인다.
* Gradient 가 작은 매개변수의 학습 속도를 늘린다.

![image](images/Optimizer_1.PNG)

----

기존 Gradient Descent 방법에서는 다음과 같은 방식으로 weight과 bias를 갱신했다.
* (weight) = (weight) - (learning rate) * (weight의 gradient)
* (bias) = (bias) - (learning rate) * (bias의 gradient)

RMSProp은 다음과 같은 방식으로 weight과 bias를 갱신한다.
* weight 과 bias 의 업데이트 속도 조절 **(클수록 업데이트 느림)**
  * $${S_{dW}} = \beta * S_{dW} + (1 - \beta)dW^2$$
  * $${S_{db}} = \beta * S_{db} + (1 - \beta)db^2$$
  * $S_{dW}$, $S_{db}$는 처음에 모두 0으로 초기화
* weight 및 bias 업데이트
  * $$(weight) = (weight) - (LearningRate) * \frac{(weight의 gradient)}{\sqrt{S_{dW} + \epsilon}}$$
  * $$(bias) = (bias) - (LearningRate) * \frac{(bias의 gradient)}{\sqrt{S_{db} + \epsilon}}$$
  * $S_{dW}$, $S_{db}$ 가 분모에 들어감으로써, weight과 bias의 업데이트 속도가 해당 값에 반비례하게 된다.

보다 자세한 설명은 다음과 같다.

* $\beta$ : 지난 몇 회의 기울기 계산의 평균치를 사용할 것인지에 대한 가중치로, 그 횟수는 $\displaystyle \frac {1}{1 - \beta}$ 이다.
  * 보통 0.9 등 1에 가까운 값 사용 
* $\epsilon$ : 계산식의 분모가 0이 되지 않게 하기 위한 작은 양수 값
  * 보통 1억 분의 1, 100억 분의 1 등의 매우 작은 값 사용

### 2-2. Adam

핵심 아이디어
* RMSProp 의 방식에 추가로 **Gradient 갱신 방향의 가속 (Momentum, 일종의 '관성' 개념)** 을 함께 이용한다.
* Momentum 은 다음의 값으로 구성된다.
  * 1차 moment $v_{dW}$ : 과거 gradient 의 지수적 가중 평균
  * 2차 moment $S_{dW}$ : 과거 gradient 의 제곱의 지수적 가중 평균

![image](images/Optimizer_2.PNG)

----

Adam은 $\alpha$, $\beta_1$, $\beta_2$, $\epsilon$ 이라는 4개의 하이퍼파라미터를 이용한다.

먼저 $v_{dW} = 0, S_{dW} = 0, v_{db} = 0, S_{db} = 0$으로 초기화한 후, 다음과 같은 계산식을 통해 최적화를 진행한다.
* $v_{dW} = \beta_1 v_{dW} + (1 - \beta_1)dW$
* $v_{db} = \beta_1 v_{db} + (1 - \beta_1)db$
* $S_{dW} = \beta_2 S_{dW} + (1 - \beta_2)dW^2$
* $S_{db} = \beta_2 S_{db} + (1 - \beta_2)db^2$

RMSProp과 달리, 다음과 같이 bias correction을 적용한다. 이는 **관련 변수들이 $v_{dW} = 0, S_{dW} = 0, v_{db} = 0, S_{db} = 0$으로 초기화** 됨에 따라 처음에 **0을 향한 편향** 이 있기 때문이다.
* $v_{dW}^{bc} = \displaystyle \frac {v_{dW}}{1 - \beta^t}$
* $v_{db}^{bc} = \displaystyle \frac {v_{db}}{1 - \beta^t}$
* $S_{dW}^{bc} = \displaystyle \frac {S_{dW}}{1 - \beta^t}$
* $S_{db}^{bc} = \displaystyle \frac {S_{db}}{1 - \beta^t}$

마지막으로 RMSProp의 가중치 갱신 방식을 유사하게 적용한다. (단, gradient 대신 bias correction 된 값을 적용한다.)
* $$(weight) = (weight) - (LearningRate) * \frac{v_{dW}^{bc}}{\sqrt{S_{dW}^{bc} + \epsilon}}$$
* $$(bias) = (bias) - (LearningRate) * \frac{v_{db}^{bc}}{\sqrt{S_{db}^{bc} + \epsilon}}$$

여기서 $\beta_1, \beta_2$는 각각 1차, 2차 moment이며, $\epsilon$은 RMSProp과 동일한 목적으로 사용한다.

### 2-3. AdamW

### 2-4. AdaGrad

### 2-5. AdaDelta

### 2-6. SGD (Stochastic Gradient Descent)

### 2-7. 기타

이 외의 Optimizer로 다음과 같은 것들이 있다.
* Nadam, Adabelief
* GD (Gradient Descent), Batch Gradient Descent
* Momentum
* NAG (Nesterov Accelerated Gradient)

## 3. 탐구: 어떤 Optimizer 가 적절할까?

### 3-1. 실험 설계

### 3-2. 실험 결과