2024.02.20 작성중

## 목차
* 1. Normalization 이란? 그 필요성은?
* 2. 다양한 Normalization 방법
  * 2-1. Min-max Normalization
  * 2-2. Standarization (Z-score normalization)
  * 2-3. Log Normalization

## Normalization 이란? 그 필요성은?
**Normalization (정규화)** 는 데이터 전처리의 방법 중 하나로, 데이터를 일정한 수식을 통해서 특정 범위로 변환하는 방법을 의미한다.

데이터 전처리 시 정규화의 필요성은 다음과 같다.
* 머신러닝 모델에 맞는 입력 및 출력 데이터
  * 데이터가 정규분포에 가까울 때 최적의 성능을 발휘하는 머신러닝 알고리즘에서는 **입력 데이터** 의 각 열을 정규화할 수 있다.
  * 특정 범위, 예를 들어 0부터 1까지의 값을 출력하는 모델의 경우, **출력 데이터** 를 정규화해야 한다.
* 데이터 feature 간 표준편차 차이에 의한 스케일 차이 해결
  * 예를 들어 두 입력 변수 A, B에 대해, 변수 A의 값의 평균이 100, 표준편차가 50이고, 변수 B의 값의 평균이 1, 표준편차가 0.25이다. 이때 변수 A, B의 값의 scale을 맞춰 주지 않으면 머신러닝 모델이 변수 A의 값에 근거하여 출력값을 결정하려는 경향이 클 것이다.
  * [kNN 머신러닝 모델에서 각 feature에 대해 Z-score 정규화를 하는 것](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Machine%20Learning%20Models/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D_%EB%AA%A8%EB%8D%B8_KNN.md) 을 예로 들 수 있다.
* outlier 의 기준을 정의했을 때, 이를 처리하기 보다 편리해진다.
  * 예를 들어 Z-score normalization (Z점수에 의한 표준화) 적용 시, [상자 수염 그림의 정의에 따라 Z 값의 절댓값이 2.7보다 큰 경우를 outlier라고 할 수 있다.](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4_%EA%B8%B0%EC%B4%88_%EC%83%81%EC%9E%90%EC%88%98%EC%97%BC%EA%B7%B8%EB%A6%BC.md)

## 다양한 Normalization 방법

### Min-max Normalization

### Standarization (Z-score normalization)

### Log Normalization