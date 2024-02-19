## 목차
1. [True Positive, True Negative, False Positive, False Negative](#True-Positive,-True-Negative,-False-Positive,-False-Negative)
2. [Accuracy, Recall, Precision](#Accuracy,-Recall,-Precision)
3. [F1 Score](#F1-Score)
4. [Type 1 Error, Type 2 Error](#Type-1-Error,-Type-2-Error)

## True Positive, True Negative, False Positive, False Negative
* True Positive (TP) : 모델의 예측이 **참**, 실제 값이 **참** 인 것
* True Negative (TN) : 모델의 예측이 **거짓**, 실제 값이 **거짓** 인 것
* False Positive (FP) : 모델의 예측이 **참**이지만, 실제 값이 **거짓** 인 것
* False Negative (FN) : 모델의 예측이 **거짓**이지만, 실제 값이 **참** 인 것

즉, 정리하면 다음 표와 같다. True/False는 **모델의 예측이 정확**한지, Positive/Negative는 **모델의 예측값**이 무엇인지를 말한다.

|예측/실제|True|False|
|---|---|---|
|True|TP|FP|
|False|FN|TN|

예를 들어, 정량품을 True, 불량품을 False로 판단하여 선별하는 것이 중요한 AI 모델을 통해 불량률을 줄여야 한다면, 실제로는 불량품(False)이지만 모델은 정량품(True)으로 인식한 케이스의 개수 (FP) 를 최대한 줄여야 한다.

## Accuracy, Recall, Precision
* Accuracy (정확도) : (TP + TN) / (TP + TN + FP + FN)
* **Recall (재현율)** : TP / (TP + **FN**)
  * **실제 값이 True**인 것들 중 모델이 True로 분류한 것의 비율
* **Precision (정밀도)** : TP / (TP + **FP**)
  * **모델이 True로 분류**한 것들 중 실제 값이 True인 것의 비율

Recall은 **False Negative**, Precision은 **False Positive**인 것을 고려한 것이다.

따라서 위에서 언급한 AI 모델을 통해 불량률을 줄여야 한다면, False Positive의 개수를 고려한 측정 지표인 Precision이 Recall보다 중요하다.

## F1 Score
* F1 Score : 2 * Precision * Recall / (Precision + Recall)

즉, Precision과 Recall의 조화 평균값이다.

## IoU
주로 Image Segmentation과 같은 task의 정확도를 구하기 위해 사용되는 값으로, 예측 영역과 실제 영역의 **(교집합의 크기) / (합집합의 크기)**, 즉 **Intersection over Union** 을 의미한다.

수식으로 나타내면 **IoU = TP / (TP + FP + FN)** 이다.

## DICE Score
**DICE Score (또는 DICE Coefficient)** 는 **(2 x 예측 영역과 실제 영역의 교집합의 크기) / (예측 영역의 크기 + 실제 영역의 크기)** 로 나타낼 수 있다.

즉, DICE Score는 다음과 같은 식으로 나타낼 수 있다.
* (2 * TP) / (2 * TP + FP + FN) = (2 * TP) / ((TP + FP) + (TP + FN))

딥러닝에서 loss를 감소하도록 학습시키기 위해서 DICE Score를 이용하여 Loss를 정의하기도 한다. DICE Score를 이용한 Loss function은 다음과 같다.
* (DICE Loss) = 1 - (DICE Score)

## 특이도 (Sensitivity)
특이도 (Sensitivity) 는 **TN / (FP + TN)** 의 값으로, 실제로 False인 데이터 중 머신러닝 모델이 False로 예측하는 비율을 의미한다.

## Type 1 Error, Type 2 Error
* Type 1 Error (1종 오류) : False Negative에 의한 오류
* Type 2 Error (2종 오류) : False Positive에 의한 오류

즉, 정리하면 다음 표와 같다.

|예측/실제|True|False|
|---|---|---|
|True||Type 2 Error|
|False|Type 1 Error||

예를 들어 직원 채용 문제에서는 다음과 같다.
* Type 1 Error (FN) : 실제로 회사에서 일하기 적합한 직원이지만, 서류, 코딩/과제 테스트, 면접 등 전형 결과 최종 불합격한 경우
* Type 2 Error (FP) : 실제로 회사에서 일하기 부적합한 직원이지만, 전형에서 최종 합격한 경우

일반적으로 Type 2 Error를 범하는 경우, 문제가 많은 직원이 최종 입사하여 회사에 큰 손해를 입힐 수 있으므로 Type 1 Error보다 회사에 피해가 크다.