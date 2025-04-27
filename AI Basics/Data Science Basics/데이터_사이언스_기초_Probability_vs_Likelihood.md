## 목차

* [1. Probability vs. Likelihood](#1-probability-vs-likelihood)
* [2. Probability (확률)](#2-probability-확률)
* [3. Likelihood (가능도, 우도)](#3-likelihood-가능도-우도)
  * [3-1. Log Likelihood (로그 가능도, 로그 우도)](#3-1-log-likelihood-로그-가능도-로그-우도)
  * [3-2. Maximum Likelihood Estimation](#3-2-maximum-likelihood-estimation)

## 1. Probability vs. Likelihood

데이터 사이언스 및 머신러닝에서 **Probability (확률)** 와 **Likelihood (가능도, 우도)** 는 둘 다 많이 쓰이는 단어이다. 이들은 얼핏 보면 비슷해 보이지만, 다음과 같은 차이점이 있다.

| 단어                   | 설명                                                  | 확률분포   | 관측값    |
|----------------------|-----------------------------------------------------|--------|--------|
| Probability (확률)     | 어떤 **확률분포** 가 주어졌을 때, 그 확률분포에서 특정 **관측값** 이 발생할 가능성 | 고정     | **변동** |
| Likelihood (가능도, 우도) | 어떤 **관측값** 이 주어졌을 때, 그 관측값이 특정 **확률분포** 에서 발생했을 가능성 | **변동** | 고정     |

즉, Probability 와 Likelihood 는 어떻게 보면 **서로 반대되는 개념** 이라고 할 수 있다.

## 2. Probability (확률)

**확률 (Probability)** 의 예시는 다음과 같다.

* 어떤 값의 확률분포가 $N(0, 1^2)$ 일 때, 이 확률분포로부터 추출한 값이 **0 이상 1 이하** 일 가능성은 **확률** 이라고 할 수 있다.
* 주사위를 50번 던져서 1의 눈이 정확히 **7번** 나올 가능성은 **확률** 이라고 할 수 있다.

## 3. Likelihood (가능도, 우도)

**가능도, 우도 (Likelihood)** 의 예시는 다음과 같다.

* **0.3521** 이라는 어떤 관측값이 주어졌을 때, 해당 관측값이 추출된 확률분포가 $N(0, 1^2)$ 일 가능성은 **Likelihood (가능도, 우도)** 이다.
* **7번** 이라는 횟수가 주어졌을 때, 해당 횟수가 **주사위를 50번 던져서 1의 눈이 나온 횟수** 일 가능성 역시 Likelihood 이다.

Likelihood (가능도) 의 수식 표현은 $L(\theta|X)$ 이고, 또한 $L(\theta|X) = Prob(X|\theta)$ 이다. 즉, **확률분포 파라미터 $\theta$ 로부터 데이터 값의 집합 $X$ 가 나타날 확률** 과 같다.

* $X$ : 데이터 값의 집합
* $\theta$ : $\theta = (\mu, \sigma)$ 로 구성되는 확률분포 (특히 정규분포) 의 파라미터

이 수식을 풀어 쓰면 다음과 같이 **각 데이터 값 $x_1, x_2, ..., x_n$ 에 대한 확률의 곱** 과 같다.

* $L(\theta|X) = Prob(X|\theta) = Prob(x_1|\theta) \times Prob(x_2|\theta) \times ... \times Prob(x_n|\theta)$

### 3-1. Log Likelihood (로그 가능도, 로그 우도)

**로그 가능도 (Log Likelihood)** 는 **가능도 (Likelihood) 의 계산 결과에 로그 (log) 를 적용** 한 것이다.

* 로그 가능도는 다음과 같은 곳에서 쓰인다.
  * 머신러닝의 [Gaussian Mixture Model (가우시안 혼합 모델)](../Machine%20Learning%20Models/머신러닝_모델_Gaussian_Mixture.md)

로그의 성질에 의해, **로그 가능도는 각 데이터 값 $x_1, x_2, ..., x_n$ 에 대한 로그 확률의 합** 이 된다.

* $L(\theta|X) = Prob(X|\theta) = Prob(x_1|\theta) \times Prob(x_2|\theta) \times ... \times Prob(x_n|\theta)$
* $log(L(\theta|X)) = log(Prob(X|\theta)) = log(Prob(x_1|\theta) \times Prob(x_2|\theta) \times ... \times Prob(x_n|\theta))$
  * $= log(Prob(x_1|\theta)) + log(Prob(x_2|\theta)) + ... + log(Prob(x_n|\theta))$

### 3-2. Maximum Likelihood Estimation

**Maximum Likelihood Estimation (MLE)** 은 **Likelihood 를 최대화하는 파라미터 $\theta$ 의 값을 찾는 것** 을 말한다.

* 이를 위해 다음과 같이 Likelihood 수식 또는 Log Likelihood 수식을 미분하여, 그 값이 0이 되는 지점을 찾는다.

| 구분                   | 수식                                                            |
|----------------------|---------------------------------------------------------------|
| Likelihood 수식 미분     | $\displaystyle \frac{\delta}{\delta \theta} L(\theta) = 0$    |
| Log Likelihood 수식 미분 | $\displaystyle \frac{\delta}{\delta \theta} ln L(\theta) = 0$ |