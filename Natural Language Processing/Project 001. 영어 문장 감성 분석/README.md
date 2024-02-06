# NLP Project 001. 영어 문장 감성 분석
* Dataset: [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=train.csv)
  * 영어로 된 트윗에 대한 감정을 분석하는 과제이다. 트윗을 한 시간, 나이, 사용자의 국가 등 추가 정보를 이용할 수 있다.
* 수행 기간: 2023.12.12, 2024.02.05 ~ 2024.02.06 (3일)

## 파일 및 코드 설명
* ```train.csv```, ```test.csv``` (데이터셋을 다운받아야 함) : 학습 및 테스트 데이터
* ```preprocess_data.csv``` : 학습 및 테스트 데이터를 전처리하는 파일
* ```run_train.py``` : 학습 데이터 기반으로 머신러닝 학습 수행 및 모델 저장
* ```run_test_using_model.py``` : 저장된 모델을 이용하여 테스트 데이터에 대해 감정 예측 및 결과 (성능지표) 분석

## 데이터 전처리 과정
* 정보 컬럼
  * ```Age_of_User``` 컬럼 : 중앙값으로 처리 (단, ```0-20``` -> ```15```, ```70-100``` -> ```80``` 으로 처리)
  * ```Time of Tweet```, ```Country``` 컬럼 : one-hot 처리 (단, train data 기준 자주 등장하는 최대 20개)
  * ```Population -2020```, ```Land Area (Km)```, ```Density (P/Km)``` 컬럼 : ```x -> log(x + 1)``` 적용한 값을 표준정규분포로 표준화
  * ```Age of User Numeric``` 컬럼 : 표준정규분포로 표준화
* 텍스트 컬럼
  * ```text``` 컬럼 : 텍스트 전처리 (웹사이트 주소 제거, ```nltk```를 이용하여 ```(```, ```)```, ```:``` tag의 단어 제거) 후 S-BERT 모델을 이용하여 임베딩
    * 사용한 S-BERT 모델 : ```paraphrase-MiniLM-L3-v2``` 트랜스포머

## 머신러닝 모델 설명
* 입력을 정보 컬럼과 텍스트 컬럼 (임베딩된) 으로 분리
* 정보 컬럼과 임베딩된 텍스트 컬럼을 각각 forwarding
* forwarding된 값들을 merge 후 forwarding 하여 최종 output 계산 (0 ~ 1, sigmoid 함수)

## 실행 순서
```
python preprocess_data.py
python run_train.py
python run_test_using_model.py
```

## 성능지표 결과
* 성능 측정지표 : ```positive=1, neutral=0.5, negative=0``` 으로 정의할 때, mean-squared-error
* 성능 측정 결과
  * **mean squared error : ```0.078267```**
  * mean absolute error : ```0.229375```
  * corr-coef between test output and ground truth : ```0.695744```

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|NLP-P1-master|||231212|240206|마스터 브랜치|
|NLP-P1-1|```done```|```feat```|231212|240205|기본 파일 추가 + 데이터 전처리 + EDA|
|NLP-P1-2|```done```|```feat```|240205|240205|모델 학습 > master|
|NLP-P1-3|```done```|```feat```|240205|240205|모델 학습 > 텍스트 전처리|
|NLP-P1-4|```done```|```feat```|240205|240206|모델 학습 > 텍스트 벡터화 (기존 NLP모델인 S-BERT 이용)|
|NLP-P1-5|```done```|```fix```|240206|240206|모델 학습 > one-hot 전처리 컬럼 선택 오류 해결|
|NLP-P1-6|```done```|```feat```|240206|240206|모델 학습 > 딥러닝 모델 개발 및 학습|
|NLP-P1-7|```done```|```feat```|240206|240206|모델 테스트 및 성능 측정|