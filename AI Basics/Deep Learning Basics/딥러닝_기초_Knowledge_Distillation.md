## 목차
* [1. Knowledge Distillation](#1-knowledge-distillation)
  * [1-1. Knowledge Distillation 은 비지도 학습임](#1-1-knowledge-distillation-은-비지도-학습임)
  * [1-2. Transfer Learning 과의 비교](#1-2-transfer-learning-과의-비교) 
* [2. Knowledge Distillation 방법의 분류](#2-knowledge-distillation-방법의-분류)
  * [2-1. Response-Based Knowledge Distillation](#2-1-response-based-knowledge-distillation)
  * [2-2. Feature-Based Knowledge Distillation](#2-2-feature-based-knowledge-distillation)
  * [2-3. Relation-Based Knowledge Distillation](#2-3-relation-based-knowledge-distillation)
* [3. Knowledge Distillation 과정](#3-knowledge-distillation-과정)
  * [3-1. KL-Divergence Loss](#3-1-kl-divergence-loss)
  * [3-2. Soft Label](#3-2-soft-label)

## 1. Knowledge Distillation

**Knowledge Distillation (지식 증류)** 는 Teacher Network 라 불리는 큰 모델로부터 **지식을 '증류' (핵심만 뽑아냄)** 하여, 이를 Student Network 라는 작은 모델로 '전달'하는 것이다.

* 이를 통해 **작은 모델이 큰 모델과 비슷한 수준의 성능을 발휘할 수 있게** 하는 것이다.

Knowledge Distillation 의 필요성은 다음과 같이 사용자 요구에 따라 **경량화된 모델이 필요** 하기 때문이다.

* 거대한 모델을 실제 제품에 적용하면, 실제 사용자 입력 데이터를 모델에 입력했을 때 모델의 출력값을 받기까지 걸리는 '추론 시간'이 길어질 수 있음
* 용량 부족으로 인해 거대한 모델 자체를 제품에 적용할 수 없고, 경량화를 시켜야 함

### 1-1. Knowledge Distillation 은 비지도 학습임

Knowledge Distillation은 **Student 모델과 Teacher 모델의 출력 분포를 유사하게 만드는** 것이 목표인 **비지도학습** 이다.

* 양쪽 모델의 출력 분포를 유사하게 만드는 데에는 **Label 이 불필요** 하다.
* 따라서, Teacher 모델의 학습 데이터를 사용할 필요가 없다.

### 1-2. Transfer Learning 과의 비교

Transfer Learning 과 Knowledge Distillation 의 차이점은 다음과 같다.

|             | Transfer Learning              | Knowledge Distillation                |
|-------------|--------------------------------|---------------------------------------|
| 양쪽 모델 간 도메인 | 서로 다름 (새로운 Domain 으로 전이 학습)    | 서로 같음 (Student 가 Teacher 의 핵심 지식을 학습) |
| 목적          | 새로운 task 의 학습 시 자원 절약          | 모델 크기 자체를 줄임                          |
| 비유          | Python 언어 지식을 이용하여 C 언어를 쉽게 학습 | Python 언어의 다양한 함수 및 문법들 중 핵심 함수/문법 학습 |

## 2. Knowledge Distillation 방법의 분류

Knowledge Distillation 방법은 **Teacher Model 로부터 정보를 얻는 방법** 에 따라 다음과 같이 3가지로 구분할 수 있다.

| 방법                                    | 설명                                                             |
|---------------------------------------|----------------------------------------------------------------|
| Response-Based Knowledge Distillation | Teacher Model 의 **output 만을 이용**한다.                            |
| Feature-Based Knowledge Distillation  | Teacher Model 의 **중간 layer 의 결과를 Student 모델에 전달**한다.           |
| Relation-Based Knowledge Distillation | Input/Hidden/Output Layer 에 있는 **feature 간 관계에 대한 정보** 를 이용한다. |

### 2-1. Response-Based Knowledge Distillation

### 2-2. Feature-Based Knowledge Distillation

### 2-3. Relation-Based Knowledge Distillation

## 3. Knowledge Distillation 과정

Knowledge Distillation 과정은 다음과 같은 특징을 갖는다.

* Distillation 을 위한 Loss Function 으로 **KL-Divergence Loss** 를 사용
  * KL-Divergence Loss 는 **두 확률분포 간의 차이** 를 나타내는 [Loss Function](딥러닝_기초_Loss_function.md) 이다.
* Hard Label 이 아닌 **Soft Label** 을 사용
  * 이를 통해 Hard Label 로 표현했을 때의 정보 손실 없이 Student Network 로 지식이 잘 전달되게 함 

### 3-1. KL-Divergence Loss

### 3-2. Soft Label

