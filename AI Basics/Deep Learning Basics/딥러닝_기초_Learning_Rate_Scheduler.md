## 목차

* [1. Learning Rate Scheduler 및 그 필요성](#1-learning-rate-scheduler-및-그-필요성)
* [2. Learning Rate Scheduler 의 종류](#2-learning-rate-scheduler-의-종류)
  * [2-1. Lambda Scheduler](#2-1-lambda-scheduler)
  * [2-2. Multiplicative Scheduler](#2-2-multiplicative-scheduler)
  * [2-3. Exponential Scheduler](#2-3-exponential-scheduler)
  * [2-4. Step Scheduler](#2-4-step-scheduler)
  * [2-5. Multi-Step Scheduler](#2-5-multi-step-scheduler)
  * [2-6. Reduce-LR-On-Plateau Scheduler](#2-6-reduce-lr-on-plateau-scheduler)
  * [2-7. Cosine-Annealing Scheduler](#2-7-cosine-annealing-scheduler)
  * [2-8. Cosine-Annealing-Warmup-Restarts Scheduler](#2-8-cosine-annealing-warmup-restarts-scheduler)
  * [2-9. Cyclic Scheduler](#2-9-cyclic-scheduler)
* [3. 실험: 가장 성능이 좋은 L.R. Scheduler 는?](#3-실험-가장-성능이-좋은-lr-scheduler-는)
  * [3-1. 실험 설계](#3-1-실험-설계)
  * [3-2. 실험 결과](#3-2-실험-결과)
  * [3-3. 실험 결과에 대한 분석](#3-3-실험-결과에-대한-이유-분석)

## 1. Learning Rate Scheduler 및 그 필요성

**Learning Rate Scheduler** 는 딥 러닝에서 **epoch가 진행됨** 에 따라 [학습률 (Learning Rate)](딥러닝_기초_Learning_Rate.md) 을 조절해 나가는 알고리즘을 의미한다.

Learning Rate Scheduler 의 필요성은 다음과 같다.

* 평균적인 gradient 가 큰 학습 초반에는 높은 Learning Rate 를 통해 빠르게 학습을 진행한다.
* 평균적인 gradient 가 작은 학습 후반에는 낮은 Learning Rate 를 통해 모델이 안정적이고 정교하게 수렴할 수 있게 한다.
* Learning rate 를 **학습 내내 일정하게 하면, 이와 같은 것을 실현할 수 없다.**

## 2. Learning Rate Scheduler 의 종류

다음과 같이 다양한 Learning Rate Scheduler 를 적절히 사용할 수 있다.

| 방법론                              | Learning Rate Scheduler                                  |
|----------------------------------|----------------------------------------------------------|
| Learning Rate 를 지수적으로 감소         | - Lambda<br>- Multiplicative<br>- Exponential            |
| Learning Rate 를 계단식 (Step) 으로 감소 | - Step<br>- Multi-Step<br>- Reduce-LR-On-Plateau         |
| 코사인 함수 그래프처럼 조정                  | - Cosine-Annealing<br>- Cosine-Annealing-Warmup-Restarts |
| 기타                               | - Cyclic<br>- One-Cycle                                  |

### 2-1. Lambda Scheduler

### 2-2. Multiplicative Scheduler

### 2-3. Exponential Scheduler

### 2-4. Step Scheduler

### 2-5. Multi-Step Scheduler

### 2-6. Reduce-LR-On-Plateau Scheduler

### 2-7. Cosine-Annealing Scheduler

### 2-8. Cosine-Annealing-Warmup-Restarts Scheduler

**1. 기본 Scheduler**

**2. Custom A: Warm-up 추가**

**3. Custom B: Max Learning Rate 지수적 감소 추가**

### 2-9. Cyclic Scheduler

## 3. 실험: 가장 성능이 좋은 L.R. Scheduler 는?

### 3-1. 실험 설계

* 모든 Scheduler 의 하이퍼파라미터는 PyTorch 에서 제공하는 기본값으로 설정

### 3-2. 실험 결과

### 3-3. 실험 결과에 대한 이유 분석