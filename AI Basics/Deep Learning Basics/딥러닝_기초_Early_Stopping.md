## 목차

* [1. Early Stopping](#1-early-stopping)
* [2. Early Stopping 의 기준 (Metric vs. Loss)](#2-early-stopping-의-기준-metric-vs-loss)
* [3. 실험 (최적 Early Stopping 기준 및 횟수)](#3-실험-최적-early-stopping-기준-및-횟수)
  * [3-1. 실험 설계](#3-1-실험-설계)
  * [3-2. 실험 결과](#3-2-실험-결과)
  * [3-3. 실험 결과에 대한 이유 분석](#3-3-실험-결과에-대한-이유-분석)

## 코드

## 1. Early Stopping

**Early Stopping** 은 딥러닝 학습 중 [overfitting](딥러닝_기초_Overfitting_Dropout.md#2-딥러닝에서의-오버피팅-overfitting) 을 방지하기 위한 방법 중 하나로, **valid dataset 의 성능지표의 최고/최저 기록이 일정 epoch 횟수 동안 갱신되지 않으면 학습을 조기 종료** 하는 것이다.

예를 들어 다음과 같다.

* **[A]** Valid set Accuracy (정확도) 의 **최고 기록**이 5 회 이상 갱신되지 않으면 학습을 조기 종료하는 경우
* **[B]** Valid Loss 의 **최저 기록**이 10 회 이상 갱신되지 않으면 학습 조기 종료

![image](images/Early_Stopping_1.PNG)

## 2. Early Stopping 의 기준 (Metric vs. Loss)

Early Stopping 의 기준으로 다음을 생각해 볼 수 있다.

* Valid data 에 대한 [Metric (Accuracy, Recall, F1 Score 등)](../Data%20Science%20Basics/데이터_사이언스_기초_Metrics.md)
* Valid data Loss

각 기준의 특징은 다음과 같다. 일반적으로는 **Accuracy나 F1 Score 등 성능지표보다는 미세한 변화를 포착할 수 있는 Loss 를 사용하는 것이 비교적 권장** 되고 있다.

| 기준             | Accuracy, F1 Score 등                                   | Loss                                                  |
|----------------|--------------------------------------------------------|-------------------------------------------------------|
| 본질             | 모델을 최종 평가하기 위한 **성능지표**                                | 이 성능지표를 올리기 위해 모델이 학습을 통해 줄여야 할 **최적화 목표**            |
| 이산/연속          | 이산적인 값                                                 | 연속적인 값                                                |
| 미세한 변화 반영      | X (계단식 변화하는 지표. valid data 개수가 많을수록 보다 작은 변화를 반영하기는 함) | O                                                     |
| overfitting 탐지 | Loss 에 비해 다소 늦을 수 있음                                   | Loss 가 감소에서 미세한 증가 추이로 변하는 것은 overfitting이 시작되었다는 의미임 |

## 3. 실험 (최적 Early Stopping 기준 및 횟수)

### 3-1. 실험 설계

### 3-2. 실험 결과

### 3-3. 실험 결과에 대한 이유 분석