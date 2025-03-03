## 목차

* [1. Loss Function 의 적절한 사용](#1-loss-function-의-적절한-사용)
  * [1-1. Multi-Label Classification 에서 Binary C.E. 를 사용하는 이유](#1-1-multi-label-classification-에서-binary-ce-를-사용하는-이유)  
  * [1-2. nn.BCELoss vs. nn.BCEWithLogitsLoss](#1-2-nnbceloss-vs-nnbcewithlogitsloss) 
* [2. 실험 설계](#2-실험-설계)
  * [2-1. 데이터셋 및 성능 Metric](#2-1-데이터셋-및-성능-metric)
  * [2-2. 실험 구성](#2-2-실험-구성)
  * [2-3. 신경망 구조](#2-3-신경망-구조)
  * [2-4. 상세 configuration](#2-4-상세-configuration)
* [3. 실험 결과](#3-실험-결과)
  * [3-1. Binary Classification](#3-1-binary-classification)
  * [3-2. Multi-Class Classification](#3-2-multi-class-classification)
  * [3-3. Multi-Label Classification](#3-3-multi-label-classification)

## 코드

* 전체 실험 코드 : [code (ipynb)](codes/Loss_Function_Misuse_experiment.ipynb)

## 1. Loss Function 의 적절한 사용

**본인이 2024년 현업 실무에서 중대한 오류를 범한 부분이라 철저히 짚고 넘어가야 한다.**

[Loss Function](딥러닝_기초_Loss_function.md) 을 잘못 사용하면 모델 학습이 잘 안 될 수 있다. Loss Function 을 적절히 사용하는 것이 중요하며, 그 방법은 다음과 같다.

| Task                              | Task 설명                                                                  | Loss Function                                                                                 |
|-----------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Regression                        |                                                                          | - MSE (Mean Squared Error)<br>- RMSE (Root Mean Squared Error)<br>- MAE (Mean Absolute Error) |
| Probability Prediction<br>(0 ~ 1) | 단일 output, 0 ~ 1 사이의 확률                                                  | 회귀 문제와 동일<br>(Task 성격이 Regression 에 가까운 경우)                                                   |
| Classification<br>(Binary)        | 각 Class 에 대한 0 ~ 1 사이의 확률<br>(Class 2개, 확률 합산은 1)                        | - BCE (Binary Cross Entropy)                                                                  |
| Classification<br>(Multi-Class)   | 각 Class 에 대한 0 ~ 1 사이의 확률<br>(Class 3개 이상, 확률 합산은 1)                     | - Categorical Cross Entropy                                                                   |
| Classification<br>(Multi-Label)   | 각 Class 에 대한 0 ~ 1 사이의 확률<br>(**각 Class 별 독립적으로 계산** 하며, 합산이 1이 아닐 수 있음) | - 각 Class 별 BCE (Binary Cross Entropy)                                                        |

### 1-1. Multi-Label Classification 에서 Binary C.E. 를 사용하는 이유

Multi-Label Classification 은 **각 Class 별 확률 값을 독립적으로 예측** 하는 것이므로, 각 Class 별로 적용할 다음과 같은 Loss Function 을 생각할 수 있다.

* Regression 에서 사용하는 Loss Function (MSE, MAE)
* Binary Cross Entropy

이 둘 중에서 더 적절한 것은 **Binary Cross Entropy** 인데, 그 이유는 다음과 같다.

* **0과 1을 반대로 한 예측에 가까울수록 페널티가 급격하게 증가** 하는 메커니즘은 MSE, MAE 등에는 없고 Cross Entropy 계열 손실 함수에만 있음
* Cross Entropy 계열 Loss Function은 0부터 1까지의 확률 값을 예측하고 그 확률을 해석하는 데 최적화되어 있음

### 1-2. nn.BCELoss vs. nn.BCEWithLogitsLoss

PyTorch 에서는 Binary Cross Entropy 함수로 **nn.BCELoss** 와 **nn.BCEWithLogitsLoss** 의 2가지 함수를 제공한다. 이들의 차이점은 다음과 같다.

| 함수                         | 설명                                                                                                                                                                                                                                                                   |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```nn.BCELoss```           | 입력되는 prediction 값을 **원래 값 그대로 해서** target 과의 Binary Cross-Entropy Loss 를 계산한다.<br>- 필요한 경우 prediction 값을 이 함수의 입력으로 넣기 위해 따로 Sigmoid 함수를 통해 0~1 범위의 값으로 변환해야 한다.                                                                                                     |
| ```nn.BCEWithLogitsLoss``` | 입력되는 prediction 값을 **[Sigmoid 함수](딥러닝_기초_활성화_함수.md#2-1-sigmoid-함수) 를 통해 0~1 로 변환시킨 값**과 target 과의 Binary Cross-Entropy Loss 를 계산한다.<br>- sigmoid 를 적용하지 않은 원본 값도 0~1 로 자동으로 변환해서 Loss 를 구하므로, 코드가 간결하고 편리하다.<br> - **이미 Sigmoid 를 통해 변환된 값을 입력으로 넣지 않도록 주의** 가 필요하다. |

![image](images/Loss_Function_4.PNG)

## 2. 실험 설계

**실험 목표**

* 다음과 같이 **부적절한 Loss Function** 을 사용했을 때, 학습 과정에서 어떤 문제가 발생하는지 알아본다.

| Task                            | Loss Function                                             |
|---------------------------------|-----------------------------------------------------------|
| Classification<br>(Binary)      | - Mean-Squared Error                                      |
| Classification<br>(Multi-Class) | - Mean-Squared Error<br> - 각 Class 별 Binary Cross-Entropy |
| Classification<br>(Multi-Label) | - Mean-Squared Error<br> - Categorical Cross-Entropy      |

### 2-1. 데이터셋 및 성능 Metric

* **MNIST Dataset (60K train / 10K test)** 에 대해, 다음과 같이 데이터를 구분한다.

| Train Data               | Valid Data                | Test Data         |
|--------------------------|---------------------------|-------------------|
| 2,000 장 (Train Set 의 일부) | 3,000 장 (Train set 에서 분리) | 10,000 장 (원본 그대로) |

* 학습 시간 절약을 위해, train dataset 중 일부만을 샘플링하여 학습
* MNIST 선정 이유
  * 데이터셋이 28 x 28 size 의 작은 이미지들로 구성
  * 이로 인해 비교적 간단한 신경망을 설계할 수 있으므로, 간단한 딥러닝 실험에 적합하다고 판단
* 성능 Metric
  * **Accuracy**
  * 선정 이유
    * Accuracy 로 성능을 측정해도 될 정도로, [각 Class 간 데이터 불균형](../Data%20Science%20Basics/데이터_사이언스_기초_데이터_불균형.md) 이 적음 

### 2-2. 실험 구성

실험에 대한 상세 구성은 다음과 같다.

| Task                            | Task 상세                                                                                                                                                                             | 실험을 진행할 잘못된 Loss Function                                |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| Classification<br>(Binary)      | 숫자를 다음과 같이 분류<br>- Class 1: 곡선 숫자 (0, 3, 6, 8, 9) 인 이미지<br>- Class 2: 나머지 숫자인 이미지                                                                                                   | - Mean-Squared Error                                     |
| Classification<br>(Multi-Class) | 0~9 의 숫자 분류, 총 10개의 Class                                                                                                                                                           | - Mean-Squared Error<br>- 각 Class 별 Binary Cross Entropy |
| Classification<br>(Multi-Label) | 숫자를 다음과 같이 4 그룹으로 나누고, 각 그룹에 속할 확률을 **독립적으로** 예측 **<br>(각 그룹에 대한 확률의 합이 1이 아닐 수 있음)**<br>- 짝수 (0, 2, 4, 6, 8)<br>- 소수 (2, 3, 5, 7)<br>- 곡선 숫자 (0, 3, 6, 8, 9)<br>- 제곱수 (0, 1, 4, 9) | - Mean-Squared Error<br> - Categorical Cross-Entropy     |

### 2-3. 신경망 구조

```python
# 신경망 구조 출력 코드

from torchinfo import summary

model = CNN()
print(summary(model, input_size=(BATCH_SIZE, 1, 28, 28)))
```

![image](images/Common_NN_Vision.PNG)

### 2-4. 상세 configuration

* [활성화 함수](딥러닝_기초_활성화_함수.md) 는 다음과 같이 사용

| Conv. Layers | Fully Connected Layer | Final Layer |
|--------------|-----------------------|-------------|
| ReLU only    | Sigmoid               | Softmax     |

* [Dropout](딥러닝_기초_Overfitting_Dropout.md#3-dropout) 미 적용
* [Early Stopping](딥러닝_기초_Early_Stopping.md) Rounds = 10 으로 고정 (10 epoch 동안 valid set 정확도 최고 기록 갱신 없으면 종료)
* Optimizer 는 [AdamW](딥러닝_기초_Optimizer.md#2-3-adamw) 를 사용
  * 해당 Optimizer 가 [동일 데이터셋을 대상으로 한 성능 실험](딥러닝_기초_Optimizer.md#3-탐구-어떤-optimizer-가-적절할까) 에서 최상의 정확도를 기록했기 때문
* Learning Rate = 0.001 로, [Learning Rate Scheduler](딥러닝_기초_Learning_Rate_Scheduler.md) 를 적용하지 않은 고정값

## 3. 실험 결과

**최종 결론**

* 권장되는 Loss Function 을 사용한 경우 모두 정상적인 수준의 성능 도출
* 권장되지 않는 Loss Function 을 사용한 경우에도, 다음과 같이 잘못된 [활성화 함수](딥러닝_기초_활성화_함수.md) 를 사용한 경우를 제외하고 정상에 가까운 성능 도출
  * Multi-Class Classification 에서 Sigmoid 함수 사용 시, 학습이 전혀 안 됨
  * Multi-Label Classification 에서 Softmax 함수 사용 시, 정상에 비해 성능이 크게 저하됨

### 3-1. Binary Classification

**Binary Cross-Entropy 대신 Mean-Squared Error 적용 시**

* 결론
  * Binary Cross-Entropy 와 Mean-Squared Error 사이에 **큰 성능 차이 없음 (오차 범위 이내로 판단)**
  * Binary Cross-Entropy 적용 시 Sigmoid 추가 적용, 최종 output 활성화 함수 (Softmax or Sigmoid) 옵션에 따른 성능 차이 역시 오차 범위 이내 
* 결론에 대한 이유 
  * 최종 output 이 Sigmoid 또는 Softmax 활성화 함수를 적용하여 0 ~ 1 의 확률로 변환된 값임
  * 결국 **Mean Squared Error** 역시 **예측 확률과 실제 Class 의 One-hot Label 간의 오차를 최소화** 하기 때문으로 추정
  * Binary Cross-Entropy 적용 시 Sigmoid 추가 적용 여부, 최종 output 활성화 함수 등은 상황에 따라 논리적으로 맞는 것을 선택해야, 최상의 성능을 볼 수 있을 것으로 기대되는 것은 물론 개발자 간 커뮤니케이션이 용이할 것으로 보임
* 성능 결과
  * BCE Loss, MSE Loss 간 성능이 큰 차이 없음

| Loss Function                                                    | Valid dataset 최고 정확도 | Test dataset 정확도 |
|------------------------------------------------------------------|----------------------|------------------|
| ✅ Binary Cross-Entropy<br>(```nn.BCELoss``` + Softmax)           | 97.10%               | 97.78%           |
| ✅ Binary Cross-Entropy<br>(```nn.BCELoss``` + Sigmoid)           | 96.67%               | 97.24%           |
| ✅ Binary Cross-Entropy<br>(```nn.BCEWithLogitsLoss``` + Softmax) | 96.77%               | 97.14%           |
| ❌ Mean-Squared Error<br>(최종 output 활성화 함수 Softmax)               | 97.00%               | 97.75%           |
| ❌ Mean-Squared Error<br>(최종 output 활성화 함수 Sigmoid)               | 97.13%               | 97.32%           |

### 3-2. Multi-Class Classification

**Categorical Cross-Entropy 대신 다른 Loss Function 적용 시**

* 결론
  * **Softmax 함수를 사용한 모든 경우** 성능이 정상적으로 나옴
    * Categorical Cross-Entropy Loss (권장되는 손실 함수) 및 Softmax + MSE, Softmax + BCE 
  * **Sigmoid 함수를 사용한 모든 경우** 아예 학습이 되지 않음
    * Sigmoid + MSE, Sigmoid + BCE 
* 결론에 대한 이유 **(추정)**

|                | Softmax + (MSE or BCE)                                                                                                                    | Sigmoid + (MSE or BCE)                                                               |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 각 출력의 분포 및 독립성 | 각 출력이 그 합이 1인 확률분포를 이루므로, MSE, BCE 등 Loss Function과 관계없이 특정 Class 로 수렴함<br>- 각 Class의 확률 값들이 상호 의존적임<br>- 따라서, **각 Class 간 경쟁 관계** 고려에 적합 | 각 출력이 독립적이며, Class 간 경쟁 관계를 고려하지 못함                                                  |
| 출력값의 해석        | Softmax의 출력값은 **해당 Class 에 속할 확률** 로 해석<br>- Multi-Class 분류 역시 출력이 각 Class 에 속할 확률로 해석 가능하므로, Softmax와 잘 조합됨                              | Sigmoid 의 출력값은 독립적임<br>- 따라서 Multi-Class 출력값의 기본적인 해석인 '각 Class에 속할 확률'을 **반영하지 못함** |

* 성능 결과
  * Categorical Cross-Entropy 성능을 포함한 모든 Softmax 성능 >> 모든 Sigmoid 성능 (학습 아예 안됨)
  * Softmax 적용 성능 간 큰 차이 없음

| Loss Function                                                    | Valid dataset 최고 정확도 | Test dataset 정확도 |
|------------------------------------------------------------------|----------------------|------------------|
| ✅ Categorical Cross-Entropy (w/ Softmax)                         | 96.83%               | 97.17%           |
| ❌ Mean-Squared Error (최종 output 활성화 함수 Softmax)                  | 96.90%               | 97.48%           |
| ❌ Mean-Squared Error (최종 output 활성화 함수 Sigmoid)                  | 10.97%               | 10.28%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCELoss``` + Softmax)           | 97.20%               | 97.39%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCELoss``` + Sigmoid)           | 10.97%               | 10.28%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCEWithLogitsLoss``` + Softmax) | 97.13%               | 97.37%           |

### 3-3. Multi-Label Classification

**정확도 (Accuracy) 산출 기준**

* 요약 : **모델이 출력하는 모든 output 에 대해, 반올림했을 때 label (0 or 1) 과 일치하는 비율**
* 각 sample 당 존재하는 ```num_classes``` (Class 종류의 개수) 개의 0 ~ 1 의 값을 prediction 으로 간주
* 모든 sample 에 대한 전체 prediction 을 가장 가까운 정수로 반올림했을 때, 정답 label 과 일치하는 비율로 정확도로 측정
  * prediction $\ge$ 0.5 인 경우 : label = 1 이면 정답
  * prediction < 0.5 인 경우 : label = 0 이면 정답

**각 Class 별 Binary Cross-Entropy 대신 다른 Loss Function 적용 시**

* 결론
  * Binary Cross Entropy (권장되는 손실 함수) 를 포함, 모든 Sigmoid 활성화 함수 사용 사례에서 정상 성능
  * 모든 Softmax 활성화 함수 사용 사례에서 다소 낮은 성능
* 결론에 대한 이유
  * Softmax 함수는 각 Class 별 확률의 합산이 1이므로, Multi-Label 에서처럼 각 Class 의 확률의 합산이 1이 넘어가거나 0에 가까운 경우를 학습하지 못함
* 성능 결과
  * 모든 Sigmoid 함수 사용 성능 > 모든 Softmax 함수 사용 성능

| Loss Function                                          | Valid dataset 최고 정확도 | Test dataset 정확도 |
|--------------------------------------------------------|----------------------|------------------|
| ✅ Binary Cross-Entropy (w/ Sigmoid)                    | 98.28%               | 98.59%           |
| ❌ Mean-Squared Error (최종 output 활성화 함수 Softmax)        | 76.34%               | 76.80%           |
| ❌ Mean-Squared Error (최종 output 활성화 함수 Sigmoid)        | 98.21%               | 98.49%           |
| ❌ Categorical Cross-Entropy (최종 output 활성화 함수 Softmax) | 79.10%               | 79.41%           |
