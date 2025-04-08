## 목차

* [1. Loss Function 의 적절한 사용](#1-loss-function-의-적절한-사용)
  * [1-1. Probability Prediction (0 ~ 1 범위 단일 output) 에서 MSE Loss 등이 부적절한 이유](#1-1-probability-prediction-0--1-범위-단일-output-에서-mse-loss-등이-부적절한-이유)
  * [1-2. Binary Classification (2 outputs) 에서 Binary C. E. 가 부적절한 이유](#1-2-binary-classification-2-outputs-에서-binary-c-e-가-부적절한-이유)
  * [1-3. Multi-Class Classification 에서 MSE Loss 등이 부적절한 이유](#1-3-multi-class-classification-에서-mse-loss-등이-부적절한-이유)
  * [1-4. Multi-Label Classification 에서 Binary C.E. 를 사용하는 이유](#1-4-multi-label-classification-에서-binary-ce-를-사용하는-이유)  
  * [1-5. nn.BCELoss vs. nn.BCEWithLogitsLoss](#1-5-nnbceloss-vs-nnbcewithlogitsloss) 
* [2. 실험 설계](#2-실험-설계)
  * [2-1. 데이터셋 및 성능 Metric](#2-1-데이터셋-및-성능-metric)
  * [2-2. 실험 구성](#2-2-실험-구성)
  * [2-3. 신경망 구조](#2-3-신경망-구조)
  * [2-4. 상세 configuration](#2-4-상세-configuration)
* [3. 실험 결과](#3-실험-결과)
  * [3-1. Regression](#3-1-regression)
  * [3-2. Probability Prediction](#3-2-probability-prediction)
  * [3-3. Binary Classification (2 outputs)](#3-3-binary-classification-2-outputs)
  * [3-4. Multi-Class Classification](#3-4-multi-class-classification)
  * [3-5. Multi-Label Classification](#3-5-multi-label-classification)
* [4. 참고 (학습 정상적 진행 여부 판단 방법)](#4-참고-학습-정상적-진행-여부-판단-방법)

## 코드

* 전체 실험 코드 : [code (ipynb)](codes/Loss_Function_Misuse_experiment.ipynb)

## 1. Loss Function 의 적절한 사용

**본인이 2024년 현업 실무에서 중대한 오류를 범한 부분이라 철저히 짚고 넘어가야 한다.**

[Loss Function](딥러닝_기초_Loss_function.md) 을 잘못 사용하면 모델 학습이 잘 안 될 수 있다. Loss Function 을 적절히 사용하는 것이 중요하며, 그 방법은 다음과 같다.

**논리적으로 부적절한 Loss Function 을 사용하는 경우, 당장 지금 있는 데이터셋에서는 성능이 잘 나오지만, 새로운 데이터셋에서는 적절한 Loss Function 을 적용했을 때보다 성능이 현저히 안 나올 수 있다.**

| Task                              | Task 설명                                                                  | Loss Function                                                                                 |
|-----------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Regression                        |                                                                          | - MSE (Mean Squared Error)<br>- RMSE (Root Mean Squared Error)<br>- MAE (Mean Absolute Error) |
| Probability Prediction<br>(0 ~ 1) | 단일 output, 0 ~ 1 사이의 확률                                                  | - Binary Cross-Entropy                                                                        |
| Classification<br>(Binary)        | 각 Class 에 대한 0 ~ 1 사이의 확률<br>(Class 2개, 확률 합산은 1)                        | - Categorical Cross Entropy                                                                   |
| Classification<br>(Multi-Class)   | 각 Class 에 대한 0 ~ 1 사이의 확률<br>(Class 3개 이상, 확률 합산은 1)                     | - Categorical Cross Entropy                                                                   |
| Classification<br>(Multi-Label)   | 각 Class 에 대한 0 ~ 1 사이의 확률<br>(**각 Class 별 독립적으로 계산** 하며, 합산이 1이 아닐 수 있음) | - 각 Class 별 BCE (Binary Cross Entropy)                                                        |

### 1-1. Probability Prediction (0 ~ 1 범위 단일 output) 에서 MSE Loss 등이 부적절한 이유

Probability Prediction 은 얼핏 보기에 **Regression Task와 유사** 하고, 따라서 MSE Loss 등을 적용해도 문제가 없을 것처럼 보인다. 그러나 **출력값이 0 ~ 1 범위 내에 있다는 점에서 본질적으로 Regression 과 차이가 있기에, MSE Loss 등은 부적절하다.**

그 이유는 다음과 같다.

* 확률 값은 0 ~ 1 에 분포해 있으며, 연속적인 실수 값이므로 MSE, MAE 등을 적용해도 성능은 그럭저럭 나올 수 있다. 그러나 **0 ~ 1 의 확률 예측에는 Cross-Entropy 계열 함수가 더 적절** 하다.
* Cross-Entropy 계열 함수에 있는 **반대되는 것에 가까운 예측** (예: 실제 특정 Class 에 해당하는데, 해당 Class 일 확률을 0.01 로 예측) 에 대해 **큰 페널티** 를 주는 메커니즘이 MSE, MAE 등에는 없다.

### 1-2. Binary Classification (2 outputs) 에서 Binary C. E. 가 부적절한 이유

결론적으로, **각 Class 별 확률을 output 으로 하는 Binary Classification 은 Class 개수가 2개인 Multi-Class Classification 과 근본적으로 동일** 하기 때문이다.

* Task 의 특성상 output 은 그 확률 값이 상호 배타적이지만, BCE Loss 는 **이들 확률 각각을 독립적으로 간주** 하기 때문에 논리적으로 부적절하다.
* Task 의 특성상 각 Class 간 배타성을 고려하는 **[Softmax 활성화 함수](딥러닝_기초_활성화_함수.md#2-5-softmax-함수)를 사용**하는데, 이는 각 확률을 독립적으로 계산하는 BCE Loss 와는 맞지 않는다.

### 1-3. Multi-Class Classification 에서 MSE Loss 등이 부적절한 이유

[One-hot Vector 화](../Machine%20Learning%20Models/머신러닝_방법론_One_Hot.md) 처리된 값을 예측함에 있어서 MSE Loss 를 사용해도 학습이 될 것 같지만 **논리적으로 부적절하다.** 그 이유는 다음과 같다.

* **MSE Loss 는 Regression Task 에 적합** 하며, 0과 1로 구성된 One-hot vector 예측의 오류를 줄이는 데는 Cross-Entropy 계열 Loss Function 에 비해 적합성이 다소 떨어진다.
* Categorical Cross Entropy 는 Multi-Class Classification task 의 특징인 **각 Class 간 확률의 배타성** 을 고려하는데, MSE 는 이런 메커니즘이 없다.

### 1-4. Multi-Label Classification 에서 Binary C.E. 를 사용하는 이유

Multi-Label Classification 은 **각 Class 별 확률 값을 독립적으로 예측** 하는 것이므로, 각 Class 별로 적용할 다음과 같은 Loss Function 을 생각할 수 있다.

* Regression 에서 사용하는 Loss Function (MSE, MAE)
* Binary Cross Entropy

이 둘 중에서 더 적절한 것은 **Binary Cross Entropy** 인데, 그 이유는 다음과 같다.

* **0과 1을 반대로 한 예측에 가까울수록 페널티가 급격하게 증가** 하는 메커니즘은 MSE, MAE 등에는 없고 Cross Entropy 계열 손실 함수에만 있음
* Cross Entropy 계열 Loss Function은 0부터 1까지의 확률 값을 예측하고 그 확률을 해석하는 데 최적화되어 있음

### 1-5. nn.BCELoss vs. nn.BCEWithLogitsLoss

PyTorch 에서는 Binary Cross Entropy 함수로 **nn.BCELoss** 와 **nn.BCEWithLogitsLoss** 의 2가지 함수를 제공한다. 이들의 차이점은 다음과 같다.

| 함수                         | 설명                                                                                                                                                                                                                                                                   |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```nn.BCELoss```           | 입력되는 prediction 값을 **원래 값 그대로 해서** target 과의 Binary Cross-Entropy Loss 를 계산한다.<br>- 필요한 경우 prediction 값을 이 함수의 입력으로 넣기 위해 따로 Sigmoid 함수를 통해 0~1 범위의 값으로 변환해야 한다.                                                                                                     |
| ```nn.BCEWithLogitsLoss``` | 입력되는 prediction 값을 **[Sigmoid 함수](딥러닝_기초_활성화_함수.md#2-1-sigmoid-함수) 를 통해 0~1 로 변환시킨 값**과 target 과의 Binary Cross-Entropy Loss 를 계산한다.<br>- sigmoid 를 적용하지 않은 원본 값도 0~1 로 자동으로 변환해서 Loss 를 구하므로, 코드가 간결하고 편리하다.<br> - **이미 Sigmoid 를 통해 변환된 값을 입력으로 넣지 않도록 주의** 가 필요하다. |

![image](images/Loss_Function_4.PNG)

## 2. 실험 설계

**실험 목표**

* 다음과 같이 **부적절한 Loss Function** 을 사용했을 때, 학습 과정에서 어떤 문제가 발생하는지 알아본다.

| Task                              | Loss Function                                                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------------------------|
| Regression                        | - Binary Cross-Entropy                                                                                    |
| Probability Prediction<br>(0 ~ 1) | - Mean-Squared Error<br>- Mean-Absolute Error<br>- Root Mean-Squared Error<br>- Categorical Cross-Entropy |
| Classification<br>(Binary)        | - Mean-Squared Error<br> - 각 Class 별 Binary Cross-Entropy                                                 |
| Classification<br>(Multi-Class)   | - Mean-Squared Error<br> - 각 Class 별 Binary Cross-Entropy                                                 |
| Classification<br>(Multi-Label)   | - Mean-Squared Error<br> - Categorical Cross-Entropy                                                      |

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
  * **Mean Squared Error (회귀), Accuracy (확률 예측 및 분류)**
  * 선정 이유
    * 분류 task의 경우, Accuracy 로 성능을 측정해도 될 정도로, [각 Class 간 데이터 불균형](../Data%20Science%20Basics/데이터_사이언스_기초_데이터_불균형.md) 이 적음 

### 2-2. 실험 구성

실험에 대한 상세 구성은 다음과 같다.

| Task                              | Task 상세                                                                                                                                                                             | 실험을 진행할 잘못된 Loss Function                                                                                 |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Regression                        | 숫자를 그 복잡도, 즉 '꺾이는 부분' 및 '교점'을 기준으로 잘랐을 때의 선 개수 + '구멍'의 개수의 값을 예측<br>- 1개 : 1<br>- 2개 : 0, 2, 3, 7<br>- 3개 : 5, 6, 9<br>- 4개 : 8<br>- 6개 : 4                                         | - Binary Cross-Entropy                                                                                    |
| Probability Prediction<br>(0 ~ 1) | 곡선 숫자 (0, 3, 6, 8, 9) 일 확률이라는 단일 output 출력                                                                                                                                          | - Mean-Squared Error<br>- Mean-Absolute Error<br>- Root Mean-Squared Error<br>- Categorical Cross-Entropy |
| Classification<br>(Binary)        | 숫자를 다음과 같이 분류하여, 각 Class 별 확률 (2개 output) 출력<br>- Class 1: 곡선 숫자 (0, 3, 6, 8, 9) 인 이미지<br>- Class 2: 나머지 숫자인 이미지                                                                    | - Mean-Squared Error<br> - 각 Class 별 Binary Cross-Entropy                                                 |
| Classification<br>(Multi-Class)   | 0~9 의 숫자 분류, 총 10개의 각 Class 별 확률 예측                                                                                                                                                 | - Mean-Squared Error<br> - 각 Class 별 Binary Cross-Entropy                                                 |
| Classification<br>(Multi-Label)   | 숫자를 다음과 같이 4 그룹으로 나누고, 각 그룹에 속할 확률을 **독립적으로** 예측 **<br>(각 그룹에 대한 확률의 합이 1이 아닐 수 있음)**<br>- 짝수 (0, 2, 4, 6, 8)<br>- 소수 (2, 3, 5, 7)<br>- 곡선 숫자 (0, 3, 6, 8, 9)<br>- 제곱수 (0, 1, 4, 9) | - Mean-Squared Error<br> - Categorical Cross-Entropy                                                      |

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

| Conv. Layers | Fully Connected Layer | Final Layer               |
|--------------|-----------------------|---------------------------|
| ReLU only    | Sigmoid               | 기본적으로 Softmax (실험에 따라 다름) |

* [Dropout](딥러닝_기초_Overfitting_Dropout.md#3-dropout) 미 적용
* [Early Stopping](딥러닝_기초_Early_Stopping.md) Rounds = 10 으로 고정 (10 epoch 동안 valid set 정확도 최고 기록 갱신 없으면 종료)
* Optimizer 는 [AdamW](딥러닝_기초_Optimizer.md#2-3-adamw) 를 사용
  * 해당 Optimizer 가 [동일 데이터셋을 대상으로 한 성능 실험](딥러닝_기초_Optimizer.md#3-탐구-어떤-optimizer-가-적절할까) 에서 최상의 정확도를 기록했기 때문
* Learning Rate = 0.001 로, [Learning Rate Scheduler](딥러닝_기초_Learning_Rate_Scheduler.md) 를 적용하지 않은 고정값

## 3. 실험 결과

**최종 결론**

* 권장되는 Loss Function 을 사용한 경우 모두 정상적인 수준의 성능 도출
* 권장되지 않는 Loss Function 을 사용한 경우에도, 다음과 같이 잘못된 [활성화 함수](딥러닝_기초_활성화_함수.md) 등을 사용한 경우를 제외하고 정상에 가까운 성능 도출
  * Regression 에서 BCE Loss 함수 사용 시, 학습이 전혀 안 되거나 런타임 오류 발생
  * Probability Prediction 및 이를 확장한 task 인 Multi-Label Classification 에서 Softmax 함수 사용 시, 성능이 크게 저하됨
  * Multi-Class Classification 에서 Sigmoid 함수 사용 시, 학습이 전혀 안 됨
  
**참고**

* 아래에서 Softmax, Sigmoid 는 별도 언급이 없으면 **최종 output 에 대한 활성화 함수** 를 이렇게 설정했음을 의미한다.

### 3-1. Regression

* 결론
  * MSE, MAE, RMSE 적용 시, 모두 정상적으로 동작하며, 성능에 큰 차이는 없다.
  * Binary Cross Entropy 실험 결과는 다음과 같다.
    * ```nn.BCELoss``` 적용 시 ```CUDA error: device-side assert triggered``` 오류 발생
    * ```nn.BCEWithLogitsLoss``` 적용 시 오차가 매우 크게 나타남
* 결론에 대한 이유
  * ```nn.BCELoss``` 에서 출력값이 0 ~ 1 의 범위 밖에 있으면, 잘못된 값 감지로 인해 오류를 발생시키는 메커니즘이 존재
  * ```nn.BCEWithLogitsLoss``` 사용 시 자체적으로 Sigmoid 가 적용되므로 모델의 출력값은 0 ~ 1 범위 안에는 있다. 이것은 실제 값의 분포 범위인 1 ~ 6 과 **큰 차이가 있으며, 모델의 출력값은 최대 1이므로 아무리 학습해도 오차를 줄일 수 없다.** 
* 성능 결과

| Loss Function                                                     | Valid dataset 최저 MSE                           | Test dataset MSE                               |
|-------------------------------------------------------------------|------------------------------------------------|------------------------------------------------|
| ✅ Mean-Squared Error<br>(출력값 정규화 O)                               | 0.0735                                         | 0.0574                                         |
| ✅ Mean-Absolute Error<br>(출력값 정규화 O)                              | 0.0799                                         | 0.0547                                         |
| ✅ Root Mean-Squared Error<br>(출력값 정규화 O)                          | 0.0667                                         | 0.0520                                         |
| ❌ Binary Cross-Entropy<br>(출력값 정규화 O, ```nn.BCELoss```)           | ```CUDA error: device-side assert triggered``` | ```CUDA error: device-side assert triggered``` |
| ❌ Binary Cross-Entropy<br>(출력값 정규화 O, ```nn.BCEWithLogitsLoss```) | **15.8230**                                    | **15.6859**                                    |

* [출력값을 정규화해야 원하는 성능이 나온다.](딥러닝_기초_활성화_함수_Misuse.md#1-2-regression-에서-정규화의-필요성) 그 방법은 학습 데이터 출력값의 평균과 표준편차를 이용하여 모든 데이터셋의 출력값에 대해 Gaussian Normalization 적용하는 것이다.
* ```nn.BCEWithLogitsLoss``` 에서 Sigmoid 가 자동 적용되는 것을 제외하고, 모든 실험 case 에서 활성화 함수 미 적용

### 3-2. Probability Prediction

* 결론
  * BCE (권장), MSE, MAE, RMSE 적용 시, 모두 정상적으로 동작하며, 성능에 큰 차이는 없다.
  * Categorical Cross-Entropy 적용 시 **학습이 아예 되지 않는다.**
  * MSE, MAE, RMSE 적용 시에는 **활성화 함수 미적용했을 때** 성능이 조금 더 좋은 듯한데, 추가 실험이 필요해 보인다.
* 결론에 대한 이유
  * 최종 output 이 0 ~ 1 의 확률로, 예측값 관점에서는 연속적인 숫자 값임
  * **MSE, MAE, RMSE** 손실 함수 역시 **예측 확률과 실제 Class 값 (0 or 1) 간의 오차를 최소화** 하기 때문
  * Categorical Cross-Entropy 적용 시에는 Softmax 함수를 사용하는데, 단일 output 값인 경우 항상 1이 되어 학습 자체가 불가능하다. 
* 성능 결과

| Loss Function                                          | Valid dataset 최고 정확도 | Test dataset 정확도 |
|--------------------------------------------------------|----------------------|------------------|
| ✅ Binary Cross-Entropy<br>(```nn.BCELoss``` + Sigmoid) | 96.30%               | 96.76%           |
| ❌ Mean-Squared Error<br>(활성화 함수 미 적용)                  | 97.43%               | 97.79%           |
| ❌ Mean-Squared Error<br>(with Sigmoid)                 | 96.57%               | 97.13%           |
| ❌ Mean-Absolute Error<br>(활성화 함수 미 적용)                 | 97.03%               | 97.88%           |
| ❌ Mean-Absolute Error<br>(with Sigmoid)                | 96.20%               | 96.54%           |
| ❌ Root Mean-Squared Error<br>(활성화 함수 미 적용)             | 98.13%               | 98.32%           |
| ❌ Root Mean-Squared Error<br>(with Sigmoid)            | 96.43%               | 97.24%           |
| ❌ Categorical Cross-Entropy<br>(with Softmax)          | **49.87%**           | **49.31%**       |

* Regression 과 달리, MSE, MAE, RMSE 를 손실 함수로 사용하는 모든 실험 case 에서 출력값 정규화 미 적용

### 3-3. Binary Classification (2 outputs)

* 결론
  * Categorical Cross-Entropy (권장), Binary Cross-Entropy, Mean-Squared Error 사이에 **큰 성능 차이 없음 (오차 범위 이내로 판단)**
  * 최종 output 활성화 함수 (Softmax or Sigmoid) 옵션에 따른 성능 차이 역시 오차 범위 이내 
* 결론에 대한 이유 
  * 최종 output 은 결국 Sigmoid 또는 Softmax 활성화 함수를 적용하여 0 ~ 1 의 확률로 변환된 값임
  * 결국 **Binary Cross-Entropy** 및 **Mean Squared Error** 손실 함수 역시 **예측 확률과 실제 Class 의 One-hot Label 간의 오차를 최소화** 하는 것이 목표임
* 성능 결과

| Loss Function                                                    | Valid dataset 최고 정확도 | Test dataset 정확도 |
|------------------------------------------------------------------|----------------------|------------------|
| ✅ Categorical Cross-Entropy<br>(with Softmax)                    | 96.57%               | 97.01%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCELoss``` + Softmax)           | 95.97%               | 96.68%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCELoss``` + Sigmoid)           | 96.50%               | 97.00%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCEWithLogitsLoss``` + Softmax) | 96.90%               | 97.12%           |
| ❌ Mean-Squared Error<br>(with Softmax)                           | 96.93%               | 97.64%           |
| ❌ Mean-Squared Error<br>(with Sigmoid)                           | 96.37%               | 96.97%           |

### 3-4. Multi-Class Classification

* 결론
  * Categorical Cross-Entropy Loss (권장) 외에도, BCE, MSE 에서 Softmax 를 사용한 경우 성능이 괜찮게 나옴 
  * **Sigmoid 함수를 사용한 모든 경우** 에서는 아예 학습이 되지 않음
* 결론에 대한 이유 **(추정)**

|                | Softmax + (Categorical CE or BCE or MSE)                                                                                                  | Sigmoid + (BCE or MSE)                                                                                                                         |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| 성능             | **정상적인 수준**                                                                                                                               | **학습이 전혀 되지 않음**                                                                                                                               |
| 각 출력의 분포 및 독립성 | 각 출력이 그 합이 1인 확률분포를 이루므로, MSE, BCE 등 Loss Function과 관계없이 특정 Class 로 수렴함<br>- 각 Class의 확률 값들이 상호 의존적임<br>- 따라서, **각 Class 간 경쟁 관계** 고려에 적합 | 각 출력이 독립적이며, Class 간 경쟁 관계를 고려하지 못함<br>- output이 2개인 Binary Classification 의 경우, **상호 배타적인 Class 가 2개에 불과** 하므로 독립성이 성능에 큰 악영향을 주지 못함 **(추정)** |
| 출력값의 해석        | Softmax의 출력값은 **해당 Class 에 속할 확률** 로 해석<br>- Multi-Class 분류 역시 출력이 각 Class 에 속할 확률로 해석 가능하므로, Softmax와 잘 조합됨                              | Sigmoid 의 출력값은 독립적임<br>- 따라서 Multi-Class 출력값의 기본적인 해석인 '각 Class에 속할 확률'을 **반영하지 못함**                                                           |

* 성능 결과

| Loss Function                                                    | Valid dataset 최고 정확도 | Test dataset 정확도 |
|------------------------------------------------------------------|----------------------|------------------|
| ✅ Categorical Cross-Entropy<br>(with Softmax)                    | 97.07%               | 97.40%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCELoss``` + Softmax)           | 97.13%               | 97.39%           |
| ❌ Binary Cross-Entropy<br>(```nn.BCELoss``` + Sigmoid)           | **11.03%**           | **10.10%**       |
| ❌ Binary Cross-Entropy<br>(```nn.BCEWithLogitsLoss``` + Softmax) | 97.33%               | 97.53%           |
| ❌ Mean-Squared Error<br>(with Softmax)                           | 97.13%               | 97.43%           |
| ❌ Mean-Squared Error<br>(with Sigmoid)                           | **10.53%**           | **11.35%**       |

### 3-5. Multi-Label Classification

**정확도 (Accuracy) 산출 기준**

* 요약 : **모델이 출력하는 모든 output 에 대해, 반올림했을 때 label (0 or 1) 과 일치하는 비율**
* 각 sample 당 존재하는 ```num_classes``` (Class 종류의 개수) 개의 0 ~ 1 의 값을 prediction 으로 간주
* 모든 sample 에 대한 전체 prediction 을 가장 가까운 정수로 반올림했을 때, 정답 label 과 일치하는 비율로 정확도로 측정
  * prediction $\ge$ 0.5 인 경우 : label = 1 이면 정답
  * prediction < 0.5 인 경우 : label = 0 이면 정답

**실험 결과**

* 결론
  * Binary Cross Entropy (권장) 를 포함, 모든 Sigmoid 활성화 함수 사용 사례에서 정상 성능
  * 모든 Softmax 활성화 함수 사용 사례에서 다소 낮은 성능
    * 이 부분에서 성능이 다소 떨어지는 원인이 **데이터셋의 난이도가 높은 것으로 추정하고 아, 그런가 보다! 하는 것은 금물!!**
* 결론에 대한 이유
  * Softmax 함수는 각 Class 별 확률의 합산이 1이므로, Multi-Label 에서처럼 각 Class 의 확률의 합산이 1이 넘어가거나 0에 가까운 경우를 제대로 학습하지 못함
* 성능 결과

| Loss Function                                 | Valid dataset 최고 정확도 | Test dataset 정확도 |
|-----------------------------------------------|----------------------|------------------|
| ✅ Binary Cross-Entropy<br>(with Sigmoid)      | 98.21%               | 98.53%           |
| ❌ Mean-Squared Error<br>(with Softmax)        | **76.43%**           | **76.74%**       |
| ❌ Mean-Squared Error<br>(with Sigmoid)        | 98.31%               | 98.49%           |
| ❌ Categorical Cross-Entropy<br>(with Softmax) | **76.57%**           | **76.95%**       |

## 4. 참고 (학습 정상적 진행 여부 판단 방법)

Valid Dataset 에 대한 Loss Function 의 값 (Valid Loss) 으로부터 **학습이 잘 되는지 (평균으로 수렴하지는 않는지)** 파악할 수 있다.

자세한 것은 [해당 문서](딥러닝_기초_Loss_function.md#5-loss-function-의-값으로-정상적-학습-진행-여부-파악) 참고.