## 목차
* [1. XGBoost (eXtreme Gradient Boosting)](#1-xgboost-extreme-gradient-boosting)
* [2. XGBoost의 장단점](#2-xgboost의-장단점)
* [3. Gradient Boosting 및 그 문제점](#3-gradient-boosting-및-그-문제점)
* [4. 알고리즘 상세](#4-알고리즘-상세)
  * [4-1. Loss Function 의 Regularization](#4-1-loss-function-의-regularization) 
  * [4-2. CART (Classification and Regression Trees)](#4-2-cart-classification-and-regression-trees)
  * [4-3. 병렬 학습을 위한 Split 알고리즘](#4-3-병렬-학습을-위한-split-알고리즘)
  * [4-4. 결측치 데이터 처리](#4-4-결측치-데이터-처리)
* [5. 하이퍼파라미터](#5-하이퍼파라미터)
  * [5-1. 기본 설정값 관련](#5-1-기본-설정값-관련) 
  * [5-2. 트리 구조 관련](#5-2-트리-구조-관련)
  * [5-3. overfitting 관련](#5-3-overfitting-관련)
  * [5-4. 기타 하이퍼파라미터](#5-4-기타-하이퍼파라미터)

## 1. XGBoost (eXtreme Gradient Boosting)
**XGBoost (eXtreme Gradient Boosting)** 은 [Ensemble (앙상블)](머신러닝_모델_Ensemble.md) 기법 중 [Boosting (부스팅)](머신러닝_모델_Ensemble.md#2-3-boosting) 방법을 적용한 모델 중 하나로, 핵심 아이디어는 다음과 같다.
* Boosting 기법 중 [Gradient Boosting](#3-gradient-boosting-및-그-문제점) 이 모델을 순차적으로 학습해야 하고, 이때 **병렬 학습이 안 된다는 점을 보완**
  * 병렬 학습을 위해, 데이터를 여러 부분 (subset) 으로 나누어 학습
* [Decision Tree](머신러닝_모델_Decision_Tree.md) 를 응용한 Regression 방식의 트리인 **CART (Classification and Regression Trees)** 를 사용

외부 링크 : [XGBoost 공식 논문](https://arxiv.org/pdf/1603.02754) 

## 2. XGBoost의 장단점
XGBoost의 장단점은 다음과 같다.

* 장점
  * 병렬 학습 가능 및 이로 인한 빠른 수행 시간
  * [Regularization](../Deep%20Learning%20Basics/딥러닝_기초_Regularization.md#l1-l2-regularization) 등을 통해 overfitting 을 방지할 수 있음
  * [자체적인 missing value (결측치) 처리 기능](#4-3-결측치-데이터-처리)
* 단점
  * 하이퍼파라미터가 많으므로 최적화에 시간이 오래 소요됨

## 3. Gradient Boosting 및 그 문제점
**Gradient Boosting** 은 Ensemble 기법 중 하나인 Boosting의 일종으로, 핵심 아이디어는 다음과 같다.
* 다음 학습할 모델로 **Residual (Loss Function의 negative gradient)** 값을 전달한다.

Residual에 대해 자세히 살펴보면 다음과 같다.

* **[Loss Function]** $L = \frac{1}{2} \times (y_i - f(x_i))^2$
  * $y_i$ : 실제 출력값
  * $x_i$ : 입력값
  * $f(x_i)$ : 모델의 예측값
* **[Gradient]** $\frac{\delta L}{\delta x_i} = f(x_i) - y_i$
* **[Residual]** $y_i - f(x_i)$ 
  * 실제 출력값과 예측값의 차이
  * $y_i - f(x_i) = -(f(x_i) - y_i)$ 이므로 이것을 **Negative Gradient** 라고 한다.

이 모델의 문제점은 **모델을 순차적으로 학습하므로 병렬 처리가 불가능** 하다는 것이다.

![image](images/XGBoost_1.PNG)

## 4. 알고리즘 상세

XGBoost에서는 다음과 같이 **CART (Classification and Regression Trees)** 와 **병렬 학습 (이를 위한 Split)** 을 적용한다.

| 구분                                                                                                                      | 목적             | 설명                                                                                                                                          |
|-------------------------------------------------------------------------------------------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Loss Function 에 [L2 와 유사한 Regularization](../Deep%20Learning%20Basics/딥러닝_기초_Regularization.md#l1-l2-regularization) 적용 | overfitting 방지 | - $Loss = \Sigma_i l(\hat(y)_i, y_i) + \Sigma_k \Omega(f_k)$<br> - $\Omega(f) = \lambda T + \frac{1}{2} \lambda w^2$                        |
| CART (Classification and Regression Trees)                                                                              |                | - **각 트리의 출력값 (연속된 숫자) 의 가중치 합** 으로 최종 예측값 결정<br>- 회귀 트리 (Regression Tree) 를 이용한 앙상블 (Ensemble)                                             |
| 병렬 학습 (Split)                                                                                                           | 모델 속도 향상       | 다음과 같이 4가지로 구분<br>- Basic Exact Greedy Algorithm<br>- Approximate Algorithm<br>- Weighted Quantile Search<br>- Sparsity-aware Split Finding |
| 결측치 데이터 처리 알고리즘 (Sparsity-aware Split Finding)                                                                          | 결측치가 있는 데이터 학습 | - **모든 결측치를 왼쪽, 오른쪽으로 몰아서** 배치했을 때의 information gain 을 각각 계산<br>- **information gain이 가장 높은** split point를 탐색                               |

### 4-1. Loss Function 의 Regularization

XGBoost의 Loss Function에는 **overfitting 방지를 위하여 다음과 같이 [L2 Regularization](../Deep%20Learning%20Basics/딥러닝_기초_Regularization.md#l1-l2-regularization) 과 유사한 Regularization** 을 적용한다.

* **[수식]** $Loss = \Sigma_i l(\hat(y)_i, y_i) + \Sigma_k \Omega(f_k)$
  * $\Omega(f) = \lambda T + \frac{1}{2} \lambda w^2$
  * $l$ : 미분 가능한 convex loss function
  * $\hat(y)_i$ : $i$ 번째 데이터에 대한 prediction
  * $y_i$ : $i$ 번째 데이터에 대한 실제 값 (target)
  * $\Omega$ : **모델 (Regression Tree 등) 이 복잡해지는 것에 대해 페널티** 를 주는 역할

이것을 논문에서는 **Regularized Learning Object** 라고 한다.

### 4-2. CART (Classification and Regression Trees)

**CART (Classification and Regression Trees)** 의 핵심 아이디어는 다음과 같다.
* leaf node에 class 대신 숫자 값이 있는 **Regression Tree** 의 최종 출력값의 가중치 합으로 최종 예측
  * Regression Tree를 이용한 Ensemble 기법으로 볼 수 있음
* Decision Tree 의 leaf node가 **Class (Categorical)** 라면, CART 의 각 Tree의 leaf node는 **Numerical** 임

![image](images/XGBoost_2.PNG)

### 4-3. 병렬 학습을 위한 Split 알고리즘

XGBoost 에서 병렬 학습을 위해 사용되는 Split 알고리즘은 다음과 같다.

| 알고리즘                                            | 핵심 아이디어                                                                                                                                |
|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| Basic Exact Greedy Algorithm                    | - 모든 feature 각각에 대해, 모든 가능한 Split point 를 탐색<br>- **Loss 감소량이 최대화** 되는 Split point 를 선택                                                |
| Approximate Algorithm                           | - 각 feature 별로 **데이터 구간을 나누고, 각 구간별로 Gaussian과 Hessian에 기반** 하여 최적의 Split point 탐색<br>- Basic Exact Greedy Algorithm 의 메모리 부족 등 문제점 보완 |
| Weighted Quantile Sketch (참고)                   | - 각 데이터에 weight이 부여되어 있을 때 사용                                                                                                          |
| [Sparsity-aware Split Finding](#4-4-결측치-데이터-처리) | - **모든 결측치를 왼쪽, 오른쪽으로 몰아서** 배치했을 때의 information gain 을 각각 계산<br>- **information gain이 가장 높은** split point를 탐색                          |

----

**1. Basic Exact Greedy Algorithm**
* 모든 feature 에 대해 모든 가능한 Split point를 탐색하고, 이들 중 **Loss 감소량이 최대화되는 것을 찾아서** Split point를 선택한다.

![image](images/XGBoost_3.PNG)

([출처](https://arxiv.org/pdf/1603.02754) : Tianqi Chen and Carlos Guestrin, XGBoost: A Scalable Tree Boosting System, 2016)

**Loss 감소량** 계산의 핵심 아이디어

* **Loss가 낮다는 것은 information (정보량) 이 높다** 는 것을 의미하므로, 그 정보량의 계산값을 score 라고 할 때 다음을 이용한다.
  * (Loss 감소량) = (전체적인 Information Gain)
  * = (양쪽 leaf node 의 score 의 합) - (부모 node 의 score)
  * = **(왼쪽 leaf node 의 score) + (오른쪽 leaf node 의 score) - (부모 node 의 score)**
* 이 score 는 1차 편미분값 Gradient 와 2차 편미분값 Hessian 을 이용하여 다음과 같이 계산한다.
  * 각 node 에 해당하는 데이터 (row) 에 대해, 다음 수식 결과의 합산
  * $\frac{(Gradient)^2}{(Hessian) + \lambda}$ 

**Loss 감소량** 의 수식

![image](images/XGBoost_4.PNG)

([출처](https://arxiv.org/pdf/1603.02754) : Tianqi Chen and Carlos Guestrin, XGBoost: A Scalable Tree Boosting System, 2016)

* $I_L$, $I_R$ : Split 이후의 left, right left node 에 속한 데이터 (row) 의 집합
* $I$ : $I = I_L \cup I_R$
* $g_i$ : $i$ 번째 데이터의 Gradient
* $h_i$ : $i$ 번째 데이터의 Hessian
* $\lambda$ : Regularization parameter 로, 전체 score 를 줄이는 역할을 함. 이를 통해 $\gamma$ 의 역할과 결합하여 overfitting 을 방지한다. 
* $\gamma$ : Information Gain이 이 값보다 작으면 분기를 중단하여 overfitting 을 방지한다.

----

**2. Approximate Algorithm**

----

### 4-4. 결측치 데이터 처리

**XGBoost 에서 결측치 처리** 를 위해서는 **Sparsity-aware Split Finding 이라는 Split 알고리즘** 이 적용된다. 그 핵심 아이디어는 다음과 같다.
* 

## 5. 하이퍼파라미터
### 5-1. 기본 설정값 관련

### 5-2. 트리 구조 관련

### 5-3. Overfitting 관련

### 5-4. 기타 하이퍼파라미터