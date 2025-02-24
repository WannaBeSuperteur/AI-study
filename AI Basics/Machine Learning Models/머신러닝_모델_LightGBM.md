## 목차
* [1. LightGBM (Light Gradient Boosting Machine)](#1-lightgbm-light-gradient-boosting-machine)
* [2. LightGBM의 장단점](#2-lightgbm의-장단점)
* [3. Leaf-wise vs. Depth-wise](#3-leaf-wise-vs-depth-wise)
* [4. 알고리즘 상세](#4-알고리즘-상세)
  * [4-1. GOSS (Gradient-based One-Side Sampling)](#4-1-goss-gradient-based-one-side-sampling)
  * [4-2. EFB (Exclusive Feature Bundling)](#4-2-efb-exclusive-feature-bundling)
* [5. 하이퍼파라미터 (Python 라이브러리 기준)](#5-하이퍼파라미터-python-라이브러리-기준)
  * [5-1. 트리 구조 관련 하이퍼파라미터](#5-1-트리-구조-관련-하이퍼파라미터)
  * [5-2. overfitting 관련 하이퍼파라미터](#5-2-overfitting-관련-하이퍼파라미터)
  * [5-3. 기타 하이퍼파라미터](#5-3-기타-하이퍼파라미터)

## 1. LightGBM (Light Gradient Boosting Machine)
**LightGBM (Light Gradient Boosting Machine)** 은 [Ensemble (앙상블)](머신러닝_모델_Ensemble.md) 기법 중 [Boosting (부스팅)](머신러닝_모델_Ensemble.md#2-3-boosting) 방법을 적용한 모델 중 하나로, 핵심 아이디어는 다음과 같다.
* [Decision Tree](머신러닝_모델_Decision_Tree.md) 기반 모델
* Tree를 균형적으로 만들기보다는 **Loss를 최소화하는 방향으로 Tree를 확장하는 leaf-wise** 알고리즘
  * 이를 통해 학습의 효율성이 향상되고, 결과적으로 학습 속도가 빨라진다. 
  * Loss를 최소화하기 위해, **data loss가 최대인 leaf node** 를 분할한다.

LightGBM은 **큰 규모의 데이터셋 (1만 개 이상의 데이터)** 에 사용하기에 보다 적합하다.

외부 링크 : [LightGBM 공식 논문](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

## 2. LightGBM의 장단점

LightGBM의 장단점은 다음과 같다.

* 장점
  * 실행 속도가 빠름 (학습 시간 짧음)
  * 메모리 사용량이 적음
  * Categorical feature 를 그대로 학습 가능하며, [One-hot Encoding](머신러닝_방법론_One_Hot.md) 보다도 성능이 좋음
    * 이는 LightGBM이 Categorical feature를 자동으로 변환하기 때문 
* 단점
  * 데이터 크기가 작은 경우 overfitting 되기 쉬움

## 3. Leaf-wise vs. Depth-wise

## 4. 알고리즘 상세
LightGBM에서는 다음과 같은 2가지 알고리즘을 사용한다.

| 알고리즘                                        | 설명                                                                                   |
|---------------------------------------------|--------------------------------------------------------------------------------------|
| **GOSS** (Gradient-based One-Side Sampling) | 값의 변화가 **Loss Function에 얼마나 큰 변화** 를 가져오는지를 기준으로, **일부 데이터만 sampling** 하여 데이터셋 크기 감소 |
| **EFB** (Exclusive Feature Bundling)        | 0이 아닌 값을 동시에 가질 확률이 매우 낮은 **상호 배타적인 feature를 bundle로 묶어서** feature 개수를 줄임            |

### 4-1. GOSS (Gradient-based One-Side Sampling)

### 4-2. EFB (Exclusive Feature Bundling)

## 5. 하이퍼파라미터 (Python 라이브러리 기준)
### 5-1. 트리 구조 관련 하이퍼파라미터

### 5-2. Overfitting 관련 하이퍼파라미터

### 5-3. 기타 하이퍼파라미터