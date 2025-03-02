## 목차
* [1. Learning Rate](#1-learning-rate)
* [2. Learning Rate 조절이 필요한 이유](#2-learning-rate-조절이-필요한-이유)
* [3. Learning Rate Scheduler](#3-learning-rate-scheduler)
* [4. Optimizer](#4-optimizer)
* [5. 탐구: 어느 정도의 Learning Rate 가 좋을까? (Accuracy vs. Time)](#5-탐구-어느-정도의-learning-rate-가-좋을까-accuracy-vs-time)
  * [5-1. 실험 설계](#5-1-실험-설계)
  * [5-2. 실험 결과](#5-2-실험-결과)

## 코드

## 1. Learning Rate

Gradient Descent 알고리즘에서는 다음과 같이 가중치를 갱신한다. ($W$ : weight) 자세한 것은 [해당 문서](../Machine%20Learning%20Models/머신러닝_모델_Linear_Logistic_Regression.md#2-2-regression-최적화-과정) 참고.

* $W = W - \alpha \frac{delta}{delta W} Loss W$

즉, [Loss Function](딥러닝_기초_Loss_function.md) 의 미분값, 즉 기울기에 **일정 배수인 $\alpha$ 를 곱한** 값만큼 갱신이 이루어진다. 여기서 $\alpha$ 를 **Learning Rate (학습률)** 라고 한다.

## 2. Learning Rate 조절이 필요한 이유

Learning Rate 조절이 필요한 이유는 다음과 같다.

* Learning Rate 가 매우 크면 학습이 수렴되지 않을 수 있다.
* Learning Rate 가 매우 작으면 학습에 시간이 매우 오래 걸린다.

따라서 이 Trade-off 를 고려한 적절한 수준의 Learning Rate 가 필요하다.

![image](images/Learning_Rate_1.PNG)

## 3. Learning Rate Scheduler

**Learning Rate Scheduler** 는 Learning Rate 를 학습이 진행됨에 따라 조절해 나가기 위한 알고리즘이다.

* 학습 초반에는 평균적인 gradient 가 크기 때문에, 높은 learning rate 를 통해 이를 빠르게 감소시킨다.
* 학습 후반에는 낮은 learning rate 를 통해 학습을 정교하게 수렴시킨다.

자세한 것은 [해당 문서](딥러닝_기초_Learning_Rate_Scheduler.md) 참고.

## 4. Optimizer

주로 Learning Rate 에 곱해지는 여러 가지 변수들을 추가하여 딥러닝 학습을 보다 효율적으로 진행하기 위한 일종의 장치이다.

자세한 것은 [해당 문서](딥러닝_기초_Optimizer.md) 참고.

## 5. 탐구: 어느 정도의 Learning Rate 가 좋을까? (Accuracy vs. Time)

### 5-1. 실험 설계

### 5-2. 실험 결과