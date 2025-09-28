
## 목차

* [1. Shared Backbone & Shared Head 개요](#1-shared-backbone--shared-head-개요)
* [2. 구조 비교](#2-구조-비교)
* [3. 탐구: 어떤 구조가 가장 좋을까?](#3-탐구-어떤-구조가-가장-좋을까)
  * [3-1. 실험 설계](#3-1-실험-설계)
  * [3-2. 실험 결과](#3-2-실험-결과)
  * [3-3. 실험 결과에 대한 이유 분석](#3-3-실험-결과에-대한-이유-분석)

## 1. Shared Backbone & Shared Head 개요

딥러닝 모델의 구조에서 **Shared Backbone** 과 **Shared Head** 는 각각 다음을 의미한다.

| 용어              | 설명                                         |
|-----------------|--------------------------------------------|
| Shared Backbone | 모델 학습 시 **여러 task 에 대해 backbone을 공유** 하는 것 |
| Shared Head     | 모델 학습 시 **여러 개의 분류/회귀 head를 공유** 하는 것      |

## 2. 구조 비교

딥러닝 모델은 Shared Backbone & Shared Head 각각의 적용 여부에 따라 다음과 같이 4가지 구조로 나눌 수 있다.

![image](images/Backbone_Head_1.PNG)

| 구조 | Shared Backbone | Shared Head |
|----|-----------------|-------------|
| 1  | X               | O           |
| 2  | X               | X           |
| 3  | O               | O           |
| 4  | O               | X           |

## 3. 탐구: 어떤 구조가 가장 좋을까?

### 3-1. 실험 설계

### 3-2. 실험 결과

### 3-3. 실험 결과에 대한 이유 분석
