## 목차
* [1. PCA (Principal Component Analysis, 주성분 분석)](#1-pca-principal-component-analysis-주성분-분석)
* [2. PCA의 기본 원리](#2-pca의-기본-원리)
* [3. PCA에서 분산이 최대인 축을 선택하는 이유](#3-pca에서-분산이-최대인-축을-선택하는-이유)
* [4. PCA의 과정](#4-pca의-과정)
  * [4-1. Covariance Matrix 계산](#4-1-covariance-matrix-계산)
  * [4-2. Eigenvector + Eigenvalue 계산](#4-2-eigenvector--eigenvalue-계산)
* [5. 탐구 (Trade-off: 모델 성능 손실 vs. 차원의 저주 해결)](#5-탐구-trade-off-모델-성능-손실-vs-차원의-저주-해결)
  * [5-1. 실험 (예시 데이터셋)](#5-1-실험-예시-데이터셋)
  * [5-2. 실험 결과](#5-2-실험-결과)

## 코드

## 1. PCA (Principal Component Analysis, 주성분 분석)
**PCA (Principal Component Analysis)** 는 데이터셋의 분산을 최대한 보존하는 "feature의 결합" 인 **주성분 (Principal Component)** 을 추출하여 **데이터셋의 차원을 축소** 하는 방법이다.
* 분산을 최대한 보존하기 위해, 데이터셋에서 **분산이 가장 큰 축 (axis)** 에 해당하는 "feature의 조합" 을 찾는다.
* [차원의 저주](../Data%20Science%20Basics/데이터_사이언스_기초_차원의_저주.md) 및 [다중공선성](../Data%20Science%20Basics/데이터_사이언스_기초_다중공선성_VIF.md)을 해결하기 위한 차원 축소의 대표적인 방법이다.

## 2. PCA의 기본 원리

## 3. PCA에서 분산이 최대인 축을 선택하는 이유

## 4. PCA의 과정
### 4-1. Covariance Matrix 계산

### 4-2. Eigenvector + Eigenvalue 계산

## 5. 탐구 (Trade-off: 모델 성능 손실 vs. 차원의 저주 해결)
* 차원의 저주에 대해서는 [해당 문서](../Data%20Science%20Basics/데이터_사이언스_기초_차원의_저주.md) 참고.

### 5-1. 실험 (예시 데이터셋)

### 5-2. 실험 결과