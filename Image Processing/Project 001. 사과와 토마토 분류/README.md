# Image Processing Project 001. 사과와 토마토 분류
* Dataset: [Apple Tomato Classification](https://www.kaggle.com/datasets/samuelcortinhas/apples-or-tomatoes-image-classification)
  * 해당 위치에서 다운받은 ```archive.zip``` 파일의 압축을 풀어서 다음과 같은 파일 데이터 구조를 만든다.

```
Project 001. 사과와 토마토 분류
- archive
  - test
    - apples
    - tomatoes
  - train
    - apples
    - tomatoes
```

* 수행 기간: 2024.02.14 ~ 2024.02.22 (9일)

## 파일 및 코드 설명
* ```augment_data.py``` : 이미지 데이터 augmentation 코드
  * 출력 파일
    * ```images/train/apples_X.png``` : ```apples``` 클래스의 학습 데이터 (원본)
    * ```images/train/apples_X_hf.png``` : ```apples``` 클래스의 학습 데이터 (```horizontal flip``` 적용)
    * ```images/train/apples_X_vf.png``` : ```apples``` 클래스의 학습 데이터 (```vertical flip``` 적용)
    * ```images/train/apples_X_af.png``` : ```apples``` 클래스의 학습 데이터 (```horizontal + vertical flip``` 적용)
    * ```tomatoes``` 클래스에 대해서도 ```apples```를 ```tomatoes```로 바꿔서 동일하게 생각하면 됨
* ```train.py``` : 모델 학습
* ```test.py``` : 모델 테스트 및 성능지표 결과 확인
* ```main.py``` : 데이터 전처리, 모델 학습, 테스트 등 전 과정을 한번에 진행

## 데이터 전처리 과정
* 이미지 데이터가 수백 장 이하로 적으므로, augmentation을 먼저 실시한다.

## 실제 학습 및 테스트 데이터
* 학습 데이터
  * ```augment_data.py``` 실행 후, ```images/train/``` 폴더 내부의 데이터
* 테스트 데이터
  * ```archive/test/apples```, ```archive/test/tomatoes``` 폴더 내부의 데이터 (원본 데이터)

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
|IP-P1-master|||240214|240222|마스터 브랜치|
|IP-P1-1|```ing```|```feat```|240215||이미지 데이터 augmentation|
|IP-P1-2||```feat```|||모델 학습|
|IP-P1-3||```feat```|||모델 테스트 및 성능지표 결과 출력|
|IP-P1-4||```feat```|||전 과정 통합 실행 코드 ```main.py``` 작성|