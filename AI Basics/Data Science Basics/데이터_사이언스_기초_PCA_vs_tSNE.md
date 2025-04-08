## 목차

* [1. PCA 와 t-SNE 의 개념](#1-pca-와-t-sne-의-개념)
* [2. PCA 와 t-SNE 의 특징 요약](#2-pca-와-t-sne-의-특징-요약)
  * [2-1. t-SNE 알고리즘 요약](#2-1-t-sne-알고리즘-요약) 
* [3. 실험 : PCA vs. t-SNE (데이터 시각화 관점)](#3-실험--pca-vs-t-sne-데이터-시각화-관점)
  * [3-1. 실험 설계](#3-1-실험-설계)
  * [3-2. 실험 결과](#3-2-실험-결과)

## 1. PCA 와 t-SNE 의 개념

**PCA (Principal Component Analysis)** 와 **t-SNE (t-distributed Stochastic Neighbor Embedding)** 의 기본 개념은 다음과 같다.

* [PCA (Principal Component Analysis)](../Machine%20Learning%20Models/머신러닝_모델_PCA.md)
  * 데이터셋의 **분산을 최대한 보존** 하는 "feature의 결합" 인 **주성분 (Principal Component)** 을 추출
  * 이를 통해 **데이터셋의 차원을 축소**
* t-SNE
  * data point 의 **상대적 거리를 유지** (멀리 떨어진 data point 는 t-SNE 결과에서도 상대적으로 멀리 떨어짐)
  * 머신러닝을 위한 차원 축소보다는 **데이터 시각화** 에 보다 중점

## 2. PCA 와 t-SNE 의 특징 요약

PCA 와 t-SNE 의 특징을 요약하면 다음과 같다.

| 구분                                 | PCA                       | t-SNE                   |
|------------------------------------|---------------------------|-------------------------|
| [차원의 저주](데이터_사이언스_기초_차원의_저주.md) 해결 | O (차원 축소)                 | O (차원 축소)               |
| 기본 목적                              | 분산을 최대한 보존 → **머신러닝에 적용** | 상대적 거리 유지 → **시각화에 중점** |
| 상대적 거리 유지                          | X                         | O                       |
| 축 계산이 항상 일정                        | O (분산이 최대인 축)             | X                       |
| 지원하는 차원                            | 몇 차원이든 지원                 | 2, 3 차원으로의 축소만 지원       |
| data point 개수 N 에 따른 연산량           |                           | $O(N^2)$                |

### 2-1. t-SNE 알고리즘 요약

* 핵심 정리
  * 고차원 feature space 에서 가까운 거리의 data point 는 **2~3 차원 feature space 에서도 거리가 가깝게** 한다.
  * [Gradient Descent](../Machine%20Learning%20Models/머신러닝_모델_Linear_Logistic_Regression.md#2-2-regression-최적화-과정) 를 이용한, 전체 data point 의 **KL Divergence (Kullback-Leibler Divergence)** 합을 최소화
* KL Divergence
  * 두 확률분포 P, Q 가 **얼마나 차이가 있는지** 를 나타내는 값
  * $\displaystyle D_{KL}(P || Q) = \Sigma_{x in X} P(x) log (\frac{P(x)}{Q(x)})$

## 3. 실험 : PCA vs. t-SNE (데이터 시각화 관점)

### 3-1. 실험 설계

### 3-2. 실험 결과