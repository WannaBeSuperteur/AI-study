# Image Processing Project 001. 사과와 토마토 분류
* Dataset: [Apple Tomato Classification](https://www.kaggle.com/code/rishavd/apple-tomato-classification)
* 수행 기간: 2024.02.14 ~ 2024.02.18 (5일)

## 파일 및 코드 설명
* ```augment_data.py``` : 이미지 데이터 augmentation 코드
* ```train.py``` : 모델 학습
* ```test.py``` : 모델 테스트 및 성능지표 결과 확인
* ```main.py``` : 데이터 전처리, 모델 학습, 테스트 등 전 과정을 한번에 진행

## 데이터 전처리 과정
* 이미지 데이터가 수백 장 이하로 적으므로, augmentation을 먼저 실시한다.

## 머신러닝 모델 설명

## 실행 순서
```
python main.py
```

## 성능지표 결과
* 성능 측정지표
  * 공통 : ```accuracy```
  * 사과, 토마토 각각 : ```precision```, ```recall```, ```F1 Score```
* 성능 측정 결과

|이미지 class|성능지표|결과|
|---|---|---|
|공통|accuracy||
|사과|F1 score||
|사과|precision||
|사과|recall||
|토마토|F1 score||
|토마토|precision||
|토마토|recall||

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|IP-P1-master|||240214|240218|마스터 브랜치|
|IP-P1-1||```feat```|||이미지 데이터 augmentation|
|IP-P1-2||```feat```|||모델 학습|
|IP-P1-3||```feat```|||모델 테스트 및 성능지표 결과 출력|
|IP-P1-4||```feat```|||전 과정 통합 실행 코드 ```main.py``` 작성|