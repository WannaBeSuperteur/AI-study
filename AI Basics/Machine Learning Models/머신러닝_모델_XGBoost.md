## 목차
* [1. XGBoost (eXtreme Gradient Boosting)](#1-xgboost-extreme-gradient-boosting)
* [2. XGBoost의 장단점](#2-xgboost의-장단점)
* [3. Gradient Boosting 및 그 문제점](#3-gradient-boosting-및-그-문제점)
* [4. 알고리즘 상세](#4-알고리즘-상세)
  * [4-1. 병렬 학습](#4-1-병렬-학습)
  * [4-2. CART (Classification and Regression Trees)](#4-2-cart-classification-and-regression-trees)
* [5. 하이퍼파라미터](#5-하이퍼파라미터)
  * [5-1. 기본 설정값 관련](#5-1-기본-설정값-관련) 
  * [5-2. 트리 구조 관련](#5-2-트리-구조-관련)
  * [5-3. overfitting 관련](#5-3-overfitting-관련)
  * [5-4. 기타 하이퍼파라미터](#5-4-기타-하이퍼파라미터)

## 1. XGBoost (eXtreme Gradient Boosting)
**XGBoost (eXtreme Gradient Boosting)** 은 [Ensemble (앙상블)](머신러닝_모델_Ensemble.md) 기법 중 [Boosting (부스팅)](머신러닝_모델_Ensemble.md#2-3-boosting) 방법을 적용한 모델 중 하나로, 핵심 아이디어는 다음과 같다.
* Boosting 기법 중 Gradient Boosting이 모델을 순차적으로 학습해야 하고, 이때 **병렬 학습이 안 된다는 점을 보완**
  * 병렬 학습을 위해, 데이터를 여러 부분 (subset) 으로 나누어 학습
* [Decision Tree](머신러닝_모델_Decision_Tree.md) 를 응용한 **CART (Classification and Regression Trees)** 를 사용

XGBoost는 sparse dataset에 적합하다.

외부 링크 : [XGBoost 공식 논문](https://arxiv.org/pdf/1603.02754) 

## 2. XGBoost의 장단점
XGBoost의 장단점은 다음과 같다.

* 장점
  * 병렬 학습 가능 및 이로 인한 빠른 수행 시간
  * [L1, L2 Regularization](../Deep%20Learning%20Basics/딥러닝_기초_Regularization.md#l1-l2-regularization) 등을 통해 overfitting 을 방지할 수 있음
  * 자체적인 missing value (결측치) 처리 기능
* 단점
  * 하이퍼파라미터가 많으므로 최적화에 시간이 오래 소요됨

## 3. Gradient Boosting 및 그 문제점
**Gradient Boosting** 은 Ensemble 기법 중 하나인 Boosting의 일종으로, 핵심 아이디어는 다음과 같다.
* 다음 학습할 모델로 값을 전달할 때, **residual (Loss Function의 negative gradient)** 를 이용한다.

## 4. 알고리즘 상세
### 4-1. 병렬 학습

### 4-2. CART (Classification and Regression Trees)

## 5. 하이퍼파라미터
### 5-1. 기본 설정값 관련

### 5-2. 트리 구조 관련

### 5-3. Overfitting 관련

### 5-4. 기타 하이퍼파라미터