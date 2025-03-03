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

| Task                            | Loss Function                                        |
|---------------------------------|------------------------------------------------------|
| Classification<br>(Binary)      | - Mean-Squared Error                                 |
| Classification<br>(Multi-Class) | - Mean-Squared Error<br> - Binary Cross-Entropy      |
| Classification<br>(Multi-Label) | - Mean-Squared Error<br> - Categorical Cross-Entropy |

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

| Task                            | Task 상세                                                                                                                                                                             | 실험을 진행할 잘못된 Loss Function                            |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| Classification<br>(Binary)      | 숫자를 다음과 같이 분류<br>- Class 1: 곡선 숫자 (0, 3, 6, 8, 9) 인 이미지<br>- Class 2: 나머지 숫자인 이미지                                                                                                   | - Mean-Squared Error                                 |
| Classification<br>(Multi-Class) | 0~9 의 숫자 분류, 총 10개의 Class                                                                                                                                                           | - Mean-Squared Error                                 |
| Classification<br>(Multi-Label) | 숫자를 다음과 같이 4 그룹으로 나누고, 각 그룹에 속할 확률을 **독립적으로** 예측 **<br>(각 그룹에 대한 확률의 합이 1이 아닐 수 있음)**<br>- 짝수 (0, 2, 4, 6, 8)<br>- 소수 (2, 3, 5, 7)<br>- 곡선 숫자 (0, 3, 6, 8, 9)<br>- 제곱수 (0, 1, 4, 9) | - Mean-Squared Error<br> - Categorical Cross-Entropy |

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

### 3-1. Binary Classification

**Binary Cross-Entropy 대신 Mean-Squared Error 적용 시**

* 결론
  * Binary Cross-Entropy 와 Mean-Squared Error 사이에 **큰 성능 차이 없음 (오차 범위 이내로 판단)**
* 결론에 대한 이유 
  * 최종 output 이 Sigmoid 또는 Softmax 활성화 함수를 적용하여 0 ~ 1 의 확률로 변환된 값임
  * 결국 **Mean Squared Error** 역시 **예측 확률과 실제 Class 의 One-hot Label 간의 오차를 최소화** 하기 때문으로 추정 
* 성능 결과
  * 3가지 Loss Function 모두 성능이 큰 차이 없음

| Loss Function                                 | Valid dataset 최고 정확도 | Test dataset 정확도 |
|-----------------------------------------------|----------------------|------------------|
| Binary Cross-Entropy                          | 96.60%               | 97.28%           |
| Mean-Squared Error (최종 output 활성화 함수 Softmax) | 97.33%               | 97.85%           |
| Mean-Squared Error (최종 output 활성화 함수 Sigmoid) | 96.93%               | 97.67%           |

### 3-2. Multi-Class Classification

**Categorical Cross-Entropy 대신 다른 Loss Function 적용 시**

* 결론
  * Categorical Cross-Entropy Loss 및 Softmax + MSE 의 경우 성능이 정상적으로 나옴
  * Sigmoid + MSE 의 경우 아예 학습이 되지 않음
* 결론에 대한 이유 **(추정)**

|                | Softmax + MSE                                                                                                         | Sigmoid + MSE                                                                        |
|----------------|-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 각 출력의 분포 및 독립성 | 각 출력이 그 합이 1인 확률분포를 이루므로, MSE 로 학습해도 특정 Class 로 수렴함<br>- 각 Class의 확률 값들이 상호 의존적임<br>- 따라서, **각 Class 간 경쟁 관계** 고려에 적합 | 각 출력이 독립적이며, Class 간 경쟁 관계를 고려하지 못함                                                  |
| 출력값의 해석        | Softmax의 출력값은 **해당 Class 에 속할 확률** 로 해석<br>- Multi-Class 분류 역시 출력이 각 Class 에 속할 확률로 해석 가능하므로, Softmax와 잘 조합됨          | Sigmoid 의 출력값은 독립적임<br>- 따라서 Multi-Class 출력값의 기본적인 해석인 '각 Class에 속할 확률'을 **반영하지 못함** |

* 성능 결과
  * Categorical Cross-Entropy 적용 시의 성능 = Softmax + MSE 성능 >> Sigmoid + MSE 성능

| Loss Function                                 | Valid dataset 최고 정확도 | Test dataset 정확도 |
|-----------------------------------------------|----------------------|------------------|
| Categorical Cross-Entropy                     | 97.07%               | 97.57%           |
| Mean-Squared Error (최종 output 활성화 함수 Softmax) | 96.87%               | 97.30%           |
| Mean-Squared Error (최종 output 활성화 함수 Sigmoid) | 10.97%               | 10.28%           |

### 3-3. Multi-Label Classification