# Meta Learning (메타러닝)

2024.02.01 작성중

## 메타 러닝이란?
**메타 러닝 (Meta Learning)** 이란 자신이 아는 것과 모르는 것을 구분한다는 개념인 '메타 인지'에 착안한 개념으로, **학습하는 방법 자체를 학습 (learn-to-learn)** 하는 개념이라고 할 수 있다.

즉, 다양한 task의 데이터의 특징 자체를 빠르게 학습하여, 새로운 데이터에 대한 학습의 성능을 빠르게 높일 수 있는 방법이다.

## 메타 러닝의 학습 방법

### Metric (거리) 기반 학습
거리 기반 학습 방법은 다음과 같은 방법을 이용한 메타 러닝이다.
* 학습 데이터 자체를 저차원의 공간으로 임베딩시킨다.
* 새로운 학습 데이터가 입력되었을 때, 그 데이터를 임베딩하여, 기존에 임베딩된 학습 데이터들과의 거리를 비교하여 가장 가까운 class로 해당 데이터를 분류한다.

### Model (모델) 기반 학습

### 최적화 학습 (Optimizer Learning)