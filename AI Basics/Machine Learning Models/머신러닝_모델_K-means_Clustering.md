## 목차
* [1. K-means Clustering 이란?](#1-k-means-clustering-이란)
* [2. K-means Clustering 알고리즘 동작 원리](#2-k-means-clustering-알고리즘-동작-원리)
* [3. K-means Clustering 알고리즘 예시](#3-k-means-clustering-알고리즘-예시)

## 1. K-means Clustering 이란?
**K-means Clustering 알고리즘** 은 데이터를 K개의 cluster (집단) 으로 나누는 비지도학습 (Unsupervised Learning) 알고리즘이다.

예를 들어 다음의 데이터 (왼쪽 그림) 를 K=3인 K-means Clustering 알고리즘을 이용하여 오른쪽 그림과 같이 분류할 수 있다.

![K-means Clustering 예시](./images/K-means_1.PNG)

이 알고리즘을 이용하여 outlier를 찾을 수도 있다.

## 2. K-means Clustering 알고리즘 동작 원리
K-means Clustering은 다음과 같은 방식으로 동작한다.

* 1. 각 feature 변수들의 값을 랜덤하게 초기화한 mean point를 K개 만든다.
* 2. 다음을 수렴할 때까지 반복한다.
  * 2-1. 데이터셋에 있는 각 data point의 class를 가장 가까운 mean point에 해당하는 class로 assign 시킨다.
  * 2-2. 각 mean point를, 전체 데이터셋에서 해당 class로 assign된 data point들의 좌표의 각 feature 별 평균값으로 갱신한다.

여기서 수렴한다는 것은, 2-1, 2-2를 한번 더 반복했을 때 data point들의 assign이 더 이상 갱신되지 않는 것을 의미한다.

## 3. K-means Clustering 알고리즘 예시
위 그림을 예로 들어서 K-means Clustering을 실시하면 다음과 같다. (K=3)

![K-means Clustering 실제 예시](./images/K-means_2.PNG)

* 초기 단계
  * **(1)** : mean point 3개를 랜덤으로 초기화한다.
* 1차 갱신
  * **(2-1) 1차** : 초기화된 mean point 중 가장 가까운 point로 각 data point의 class를 assign한다.
  * **(2-2) 1차** : mean point를 해당 class의 모든 data point들의 각 좌표 (feature 값) 의 평균값으로 갱신한다. (각 feature별로)
* 2차 갱신
  * **(2-1) 2차** : green class의 점 2개를 blue class로 re-assign한다.
  * **(2-2) 2차** : re-assign이 발생한 class인 green, blue의 mean point를 각각 갱신한다.
* 3차 갱신
  * **(2-1) 3차** : green class의 점 2개를 blue class로, blue class의 점 1개를 yellow class로 각각 re-assign한다.
  * **(2-2) 3차** : 모든 class에 re-assign (해당 class의 점 추가 또는 기존 점이 다른 class로 이동) 이 발생했으므로, 모든 mean point를 갱신한다.
* 4차 갱신
  * **(2-1) 4차** : blue -> yellow (점 1개), yellow -> green (점 1개) 로 data point의 class를 갱신한다.
  * **(2-2) 4차** : 모든 class에 re-assign이 발생했으므로, 모든 mean point를 갱신한다.
* 최종 결과
  * **최종 결과** : 이후 re-assign은 더 이상 일어나지 않으므로, 이것이 최종 수렴한 K-means Clustering의 결과이다.