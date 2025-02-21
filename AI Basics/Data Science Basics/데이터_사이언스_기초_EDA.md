## 목차
* [1. EDA란?](#1-eda란)
* [2. EDA의 목적](#2-eda의-목적)
* [3. EDA의 방법](#3-eda의-방법)
* [4. 실제 EDA 예시](#4-실제-eda-예시)
* [5. 라이브러리 선택 (Matplotlib vs. Seaborn vs. Plotly)](#5-라이브러리-선택-matplotlib-vs-seaborn-vs-plotly)

## 1. EDA란?
**EDA (탐색적 데이터 분석, Exploratory Data Analysis)** 는 데이터를 이용하여 본격적으로 모델링을 시작하기 전에, 데이터를 **직관적으로 파악**하기 위한 과정을 말한다.

## 2. EDA의 목적
* 데이터의 전반적인 값의 분포를 찾아내고, 이에 따른 이상치 (outlier) 등을 찾아낸다.
* 데이터에 있는 여러 feature 중 필요한 feature들과 불필요한 feature들을 찾아내서, 필요한 feature들을 이후에 집중적으로 분석한다.

## 3. EDA의 방법
* 각 변수 간의 상관계수를 이용하여 상관관계 (correlation) 를 파악한다.
  * 이를 통해, target 변수를 포함한 다른 모든 변수와 상관관계가 매우 낮은 (상관계수가 0에 가까운) 변수는 무의미한 변수에 가깝다고 할 수 있다.
  * 변수들 간 상관관계가 존재하는 다중공선성 (multicollinearity) 을 추가로 파악해야 할 수 있다.
  * 변수 간 상관관계를 더 정확히 파악하기 위해 scatter plot (산점도) 을 활용할 수 있다.
* 각 변수의 분포를 확인한다.
  * 각 변수별로 상자 수염 그림 (box and whisker plot) 을 이용하여, 값의 분포 및 outlier 등을 분석할 수 있다.
* K-means clustering과 같은 머신러닝 알고리즘을 이용하여 outlier를 찾는다.

## 4. 실제 EDA 예시
일반적으로 다음과 같은 것들을 EDA의 대상으로 삼는다.
* 단일 feature 에 대해, 그 값의 분포
  * Numeric, Categorical, ...
  * [해당 문서](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/데이터_사이언스_기초_EDA_단일_feature_분포.md) 참고. 
* feature 간의 상관관계
  * Numeric features, Categorical features, Numerical vs. Categorical, ...
  * [해당 문서](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Data%20Science%20Basics/데이터_사이언스_기초_EDA_feature_상관관계.md) 참고. 

## 5. 라이브러리 선택 (Matplotlib vs. Seaborn vs. Plotly)
* 결론 : **Plotly (1순위) > Seaborn (2순위) > Matplotlib (3순위)**
  * 결과물 (그래프 등) 의 디자인이 Plotly가 다른 라이브러리에 비해 깔끔함
  * Plotly는 다른 라이브러리에 없는 다음과 같은 특징 존재
    * 마우스를 hover 하면 자동으로 값 표시 (실무적으로, 보고서 작성 시 정확한 값을 알아야 할 때 중요)
  * Seaborn은 **KDE (Kernel Density Estimation)** 그래프를 표시할 수 있다는 장점이 있으므로, 경우에 따라 Plotly와 병행 사용 추천
* 상세 근거는 "4. 실제 EDA 예시" 에 링크된 각 문서의 **하단에 있는 "탐구" 문단** 참고