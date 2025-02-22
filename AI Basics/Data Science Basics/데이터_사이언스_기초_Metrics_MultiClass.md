## 목차
Accuracy, Recall, Precision 등 Metric에 대한 기본 설명은 [해당 문서](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/데이터_사이언스_기초_Metrics.md) 참고.

* [1. Multi-Class Classification Problem (다중 분류 문제)](#1-multi-class-classification-problem-다중-분류-문제)
* [2. Confusion Matrix](#2-confusion-matrix)
* [3. Recall, Precision](#3-recall-precision)
* [4. F1 Score](#4-f1-score)
  * [4-1. Macro F1 Score](#4-1-macro-f1-score) 
  * [4-2. Micro F1 Score](#4-2-micro-f1-score)
  * [4-3. Weighted F1 Score](#4-3-weighted-f1-score)
  * [4-4. Accuracy = Micro F1 Score = Micro Precision = Micro Recall 증명](#4-4-accuracy--micro-f1-score--micro-precision--micro-recall-증명)
* [5. 탐구: Macro, Micro, Weighted F1 Score는 언제, 왜 사용하는가?](#5-탐구-macro-micro-weighted-f1-score는-언제-왜-사용하는가)
* [6. 탐구: Micro F1 Score = Accuracy 인데 굳이 구분하는 이유는?](#6-탐구-micro-f1-score--accuracy-인데-굳이-구분하는-이유는)

## 1. Multi-Class Classification Problem (다중 분류 문제)
* 이미지를 **개, 고양이** 라는 2개의 class로 분류하는 것이 아닌, **개, 고양이, 말, 호랑이, 뱀, ...** 과 같이 3개 이상의 class로 분류해야 하는 머신러닝 task가 있을 수 있다.
* 여기서는 이와 같이 **3개 이상** 의 class에 대한 **다중 분류** 문제에 대해 다룬다.

## 2. Confusion Matrix
* Multi-Class 문제에서의 Confusion Matrix는 다음과 같이 **Positive / Negative 대신 각 Class를 사용한다.**

|              | 실제 값 =<br>Class A | 실제 값 =<br>Class B | 실제 값 =<br>Class C | Preicsion  |
|--------------|-------------------|-------------------|-------------------|------------|
| 예측 = Class A | (정답)              |                   |                   |            |
| 예측 = Class B |                   | (정답)              |                   |            |
| 예측 = Class C |                   |                   | (정답)              |            |
| Recall       |                   |                   |                   | (Accuracy) |  

여기서는 다음과 같은 예시를 사용한다.

|          | 실제 값 = 개 | 실제 값 = 고양이 | 실제 값 = 뱀 | Preicsion          |
|----------|----------|------------|----------|--------------------|
| 예측 = 개   | **9**    | 0          | 2        | 81.8%              |
| 예측 = 고양이 | 0        | **6**      | 1        | 85.7%              |
| 예측 = 뱀   | 1        | 4          | **17**   | 77.3%              |
| Recall   | 90.0%    | 60.0%      | 85.0%    | (Accuracy = 80.0%) |  

## 3. Recall, Precision
* **Recall (재현율)** : 각 class 별로 다음과 같이 원래 수식 그대로 **TP / (TP + FN)** 으로 계산
  * 해당 class 인 경우 Positive, 다른 class 인 경우 Negative 로 간주
* **Precision (정밀도)** : 각 class 별로 다음과 같이 원래 수식 그대로 **TP / (TP + FP)** 로 계산
  * 해당 class 인 경우 Positive, 다른 class 인 경우 Negative 로 간주

위 예시에서 Recall과 Precision은 각각 다음과 같이 계산한다.

| 구분        | Class | 계산                            | 의미                             |
|-----------|-------|-------------------------------|--------------------------------|
| Recall    | 개     | 9 / (9 + 0 + 1) = **90.0%**   | 실제 '개'를 '개'로 예측한 비율            |
| Recall    | 고양이   | 6 / (0 + 6 + 4) = **60.0%**   | 실제 '고양이'를 '고양이'로 예측한 비율        |
| Recall    | 뱀     | 17 / (2 + 1 + 17) = **85.0%** | 실제 '뱀'을 '뱀'으로 예측한 비율           |
| Precision | 개     | 9 / (9 + 0 + 2) = **81.8%**   | '개'로 예측한 것 중 실제 '개'인 것의 비율     |
| Precision | 고양이   | 6 / (0 + 6 + 1) = **85.7%**   | '고양이'로 예측한 것 중 실제 '고양이'인 것의 비율 |
| Precision | 뱀     | 17 / (1 + 4 + 17) = **77.3%** | '뱀'으로 예측한 것 중 실제 '뱀'인 것의 비율    |

## 4. F1 Score
Multi-Class에서의 F1 Score는 다음과 같이 3가지로 구분된다.
* **Macro F1 Score**
  * 모든 class에 대해 **동일한 가중치** 를 적용한 F1 Score의 단순 평균
* **Micro F1 Score**
  * 모든 class의 TP, FP, FN 의 값을 합산하여, 이 값을 이용하여 구한 F1 Score
* **Weighted F1 Score**
  * 각 class에 대해 **해당 class의 실제 데이터 개수만큼 가중치** 를 적용한 F1 Score

## 4-1. Macro F1 Score
**Macro F1 Score** 를 구하는 방법은 다음과 같다.
* 각 Class 에 대한 Recall 과 Precision 을 구한다.
* 이를 이용하여 각 Class 에 대한 F1 Score 를 구한다.
* **이렇게 구한 F1 Score 를 가중치 없이 단순 평균** 한다.

위 예시에서는 다음과 같이 계산한다.

**1. 각 Class에 대한 Recall과 Precision 계산**

| Class | Recall | Precision |
|-------|--------|-----------|
| 개     | 0.900  | 0.818     |
| 고양이   | 0.600  | 0.857     |
| 뱀     | 0.850  | 0.773     |

**2. 각 Class에 대한 F1 Score 계산**

| Class | Recall | Precision | F1 Score                                        |
|-------|--------|-----------|-------------------------------------------------|
| 개     | 0.900  | 0.818     | **0.857** = 2 * 0.900 * 0.818 / (0.900 + 0.818) |
| 고양이   | 0.600  | 0.857     | **0.706** = 2 * 0.600 * 0.857 / (0.600 + 0.857) |
| 뱀     | 0.850  | 0.773     | **0.810** = 2 * 0.850 * 0.773 / (0.850 + 0.773) |

**3. F1 Score를 가중치 없이 단순 평균**

* (Macro F1 Score) = (0.857 + 0.706 + 0.810) / 3 = **0.791 (79.1%)**

## 4-2. Micro F1 Score
**Micro F1 Score** 를 구하는 방법은 다음과 같다.
* 각 Class에 대해 측정된 True Positive (TP), False Positive (FP), False Negative (FN) 을 모두 합산한다.
* 합산된 TP, FP, FN 에 대해 Precision (Micro Precision) 과 Recall (Micro Recall) 을 계산한다.
* **Micro Precision과 Micro Recall을 이용하여 Micro F1 Score를 계산한다.**

위 예시에서는 다음과 같이 계산한다.

**1. 각 Class에 대해 TP, FP, FN의 개수 정리 및 합산**

| Class | TP     | FP    | FN    |
|-------|--------|-------|-------|
| 개     | 9      | 2     | 1     |
| 고양이   | 6      | 1     | 4     |
| 뱀     | 17     | 5     | 3     |
| 합계    | **32** | **8** | **8** |

**2. 합산된 TP, FP, FN에 대해 Micro Precision 및 Micro Recall 계산**

* Micro Precision = TP / (TP + FP) = 32 / (32 + 8) = **0.800**
* Micro Recall = TP / (TP + FN) = 32 / (32 + 8) = **0.800**
* Micro F1 Score = 2 * (Micro Precision) * (Micro Recall) / ((Micro Precision) + (Micro Recall)) = 2 * 0.800 * 0.800 / (0.800 + 0.800) = **0.800 (80.0%)**

## 4-3. Weighted F1 Score
**Weighted F1 Score** 를 구하는 방법은 다음과 같다.
* Macro F1 Score 를 구하는 방법과 같지만, **F1 Score를 각 class의 실제 데이터 개수만큼 가중치** 를 둔다.

위 예시에서는 다음과 같이 계산한다.

**1. 각 Class에 대한 F1 Score 계산 및 해당 class의 실제 데이터 개수 정리**

| Class | Recall | Precision | F1 Score                                        | 실제 데이터 개수 |
|-------|--------|-----------|-------------------------------------------------|-----------|
| 개     | 0.900  | 0.818     | **0.857** = 2 * 0.900 * 0.818 / (0.900 + 0.818) | 10        |
| 고양이   | 0.600  | 0.857     | **0.706** = 2 * 0.600 * 0.857 / (0.600 + 0.857) | 10        |
| 뱀     | 0.850  | 0.773     | **0.810** = 2 * 0.850 * 0.773 / (0.850 + 0.773) | 20        |

**2. F1 Score를 각 Class 별 실제 데이터 개수를 가중치로 하여 계산**

* (Weighted F1 Score) = (**10** * 0.857 + **10** * 0.706 + **20** * 0.810) / (10 + 10 + 20) = **0.796 (79.6%)**

## 4-4. Accuracy = Micro F1 Score = Micro Precision = Micro Recall 증명
위 실험 결과에서 Accuracy, Micro F1 Score, Micro Precision, Micro Recall의 값이 모두 같게 나와서 의문이 들어서, 이를 증명하려고 한다.

결론적으로, 다음이 성립한다.
* 임의의 Multi-Class 데이터셋 테스트 결과에 대해서, **Micro F1 Score = Micro Precision = Micro Recall** 이 성립한다.
* 임의의 Multi-Class 데이터셋 테스트 결과에 대해서, **Accuracy = Micro F1 Score** 가 성립한다.

여기서는 다음과 같은 notation 을 이용한다.
* C1, C2, ..., CN : 각각의 class
* count(CA,CB) : 예측은 CA (class A), 실제는 CB (class B) 인 데이터의 개수. A = B 일 수 있음.

**1. Micro F1 Score = Micro Precision = Micro Recall 증명**

먼저, TP, FP, FN 개수의 합산을 구하면 다음과 같다.
* TP 개수의 합산 = count(C1,C1) + count(C2,C2) + ... + count(CN,CN)
  * = (Confusion Matrix 에서 **정답**을 나타내는, 주대각선 성분의 합) 

* FP 개수의 합산 = (C1에 대한 FP) + ... + (CN에 대한 FP)
  * = (예측은 C1이지만 실제 값은 C1이 아님) + ... + (예측은 CN이지만 실제 값은 CN이 아님) 
  * = { count(C1,C2) + ... + count(C1,CN) } + ... + { count(CN,C1) + ... + count(CN, C(N-1)) }
  * = (Confusion Matrix 에서 **오답**을 나타내는, 주대각선 외의 모든 성분의 합)

* FN 개수의 합산 = (C1에 대한 FN) + ... + (CN에 대한 FN)
  * = (예측은 C1이 아니지만 실제 값은 CN) + ... + (예측은 CN이 아니지만 실제 값은 CN)
  * = { count(C2,C1) + ... + count(CN,C1) } + ... + { count(C1,CN) + ... + count(C(N-1), CN) }
  * = (Confusion Matrix 에서 **오답**을 나타내는, 주대각선 외의 모든 성분의 합)

* 따라서 **(FP 개수의 합산) = (FN 개수의 합산) = (전체에서 오답을 한 데이터 개수)** 라고 할 수 있다.

이제 Micro Precision, Micro Recall, Micro F1 Score를 계산하면 다음과 같다.
* (Micro Precision) = (TP 개수의 합산) / ((TP 개수의 합산) + (FP 개수의 합산))
  * = (전체에서 정답 개수) / ((전체에서 정답 개수) + (전체에서 오답 개수))

* (Micro Recall) = (TP 개수의 합산) / ((TP 개수의 합산) + (FN 개수의 합산))
  * = (전체에서 정답 개수) / ((전체에서 정답 개수) + (전체에서 오답 개수))
  * = Micro Precision

* (Micro F1 Score) = 2 * (Micro Precision) * (Micro Recall) / ((Micro Precision) + (Micro Recall))
  * = 2 * (Micro Precision) * (Micro Precision) / ((Micro Precision) + (Micro Precision))
  * Micro Precision

따라서 **Micro F1 Score = Micro Precision = Micro Recall** 은 항상 성립한다.

**2. Accuracy = Micro F1 Score 증명**

위 결과에서 다음을 알 수 있다.
* Micro F1 Score = Micro Precision
* Micro Precision = (전체에서 정답 개수) / ((전체에서 정답 개수) + (전체에서 오답 개수))

한편, Accuracy의 정의에 의해 다음이 성립한다.
* Accuracy = (전체에서 정답 개수) / ((전체에서 정답 개수) + (전체에서 오답 개수))

따라서 **Accracy = Micro Precision = Micro F1 Score** 가 항상 성립한다.

## 5. 탐구: Macro, Micro, Weighted F1 Score는 언제, 왜 사용하는가?
요약하면 다음과 같다.

| 구분                | 사용하는 경우                                                                                                                  | 사용 이유                                                                       |
|-------------------|--------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Macro F1 Score    | 클래스 간 [데이터 불균형](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/데이터_불균형.md) 시 | - 모든 class에 대한 예측 성능을 균일하게 평가할 때<br>- 데이터 개수가 적은 클래스에 대한 예측 성능도 중요하다고 판단될 때 |
| Micro F1 Score    | 모델의 전반적인 성능 평가 시                                                                                                         |                                                                             |
| Weighted F1 Score | 클래스 별 데이터 개수를 고려한 성능 평가 시                                                                                                |                                                                             |

예를 들어, 다음과 같은 Confusion Matrix에 대해 Macro / Micro / Weighted F1 Score는 다음과 같다.

|          | 실제 값 = 정량품 | 실제 값 = 불량A | 실제 값 = 불량B | Preicsion          |
|----------|------------|------------|------------|--------------------|
| 예측 = 정량품 | **880**    | 5          | 15         | 97.8%              |
| 예측 = 불량A | 8          | **10**     | 2          | 50.0%              |
| 예측 = 불량B | 21         | 4          | **55**     | 68.8%              |
| Recall   | 96.8%      | 52.6%      | 76.4%      | (Accuracy = 94.5%) |  

* 각 Class에 대한 F1 Score
  * 정량품 : **97.3%**
  * 불량A : **51.3%**
  * 불량B : **72.4%**
* 각 F1 Score
  * Macro F1 Score : **73.6%**
  * Micro F1 Score [(=Accuracy)](#4-4-accuracy--micro-f1-score--micro-precision--micro-recall-증명) : **94.5%**
  * Weighted F1 Score : **94.4%**
* 여기서는 각 F1 Score를 다음과 같은 경우에 사용하면 적절하다.

|구분| 사용하는 경우                                                                                                                                  |
|---|------------------------------------------------------------------------------------------------------------------------------------------|
|Macro F1 Score| 불량A, 불량B 등 불량품을 모델이 불량품으로 예측하는 것도 **정량품을 예측하는 것만큼 중요할 때** (데이터 불균형이 있지만 불량품 Class에 대한 성능 역시 중요함)<br>- 이 상황에서 불량률을 줄여야 할 때 사용하기에 적합한 성능지표 |
|Micro F1 Score| 정량품을 잘 분류하는 것을 포함하여 모델의 전반적인 성능을 측정할 때 (서로 동일한 값임이 확인된 Accuracy 와 유사한 목적)                                                                |

## 6. 탐구: Micro F1 Score = Accuracy 인데 굳이 구분하는 이유는?
결론적으로, 성능 평가에서 각각 서로 다른 측면과 일관성을 강조하기 위한 **커뮤니케이션 상의 이유** 로 추정된다.

* 성능 평가에서 서로 다른 측면 강조
  * Accuracy : **모델의 전반적인 정확도**를 의미
  * Micro F1 Score : Micro F1 Score도 결국 **Precision과 Recall을 조화 평균한 F1 Score**를 의미하므로, **Precision과 Recall의 측면**에서의 모델의 성능을 의미

* 일관성 유지
  * Macro F1 Score, Weighted F1 Score 를 측정해야 하는 경우, 'Micro F1 Score'라는 이름을 사용함으로써 본질적으로 유사한 F1 Score 를 이용한 성능 측정 결과를 비교 분석 가능 
