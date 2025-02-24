## 목차
* [1. Random Forest](#1-random-forest)
* [2. Random Forest 모델의 장단점](#2-random-forest-모델의-장단점)
* [3. Random Forest 모델 평가: Out-of-bag (OOB)](#3-random-forest-모델-평가-out-of-bag-oob)

## 1. Random Forest
**Random Forest** 의 핵심 아이디어는 다음과 같다.
* 서로 다른 [Decision Tree](머신러닝_모델_Decision_Tree.md) 를 **데이터 샘플 및 feature를 랜덤하게 선택** 하여 여러 개 생성
  * Decision **Tree** 가 여러 개이기 때문에 Random **Forest** 라고 부름 
* 이들 Decision Tree에 대한 [Bagging 방식의 Ensemble](머신러닝_모델_Ensemble.md#2-2-bagging) 기법이다.

즉, **Random Forest = Decision Tree + Bagging (또는 Ensemble)** 이라고 할 수 있다.

## 2. Random Forest 모델의 장단점

## 3. Random Forest 모델 평가: Out-of-bag (OOB)
