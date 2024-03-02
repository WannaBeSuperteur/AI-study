# Generative AI 프로젝트 1. MNIST 숫자 생성 AI with C-VAE
* Dataset: [MNIST DataSet](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data)
  * 위 주소에서 다운로드 받은 ```mnist_train.csv``` 파일이 학습 데이터이다.
* 수행 기간: 2024.03.02 ~ 2024.03.10 (9일)

## 파일 및 코드 설명
* ```mnist_train.csv``` : 학습 데이터셋
* ```train.py``` : 데이터 전처리를 포함한 학습 과정 전체 진행
  * 필요 파일 : ```mnist_train.csv``` (학습 데이터셋)
  * 출력 모델 : ```cvae_model``` (Conditional VAE 모델)
* ```test.py``` : Class 가 주어지면 해당 Class 를 이용하여 이미지 생성
  * 필요 모델 : ```cvae_encoder_model``` (인코더), ```cvae_decoder_model``` (디코더), ```cvae_model``` (전체 모델)
  * 출력 파일 : ```test_output.png```

## 데이터 전처리 및 생성 과정
* ```minst_train.csv``` 파일의 각 행을 class (1) 와 image 부분 (784 = 28 x 28) 으로 분리

## 머신러닝 모델 설명
* Conditional VAE 모델을 이용
  * VAE 참고 : [Variational Auto-Encoder](https://github.com/WannaBeSuperteur/AI-study/blob/main/Generative%20AI/Basics_Variational%20Auto%20Encoder.md)
  * Conditional VAE (C-VAE) 는 VAE를 통해 생성하려는 데이터에 **제약 조건** 을 추가한 형태의 모델
  * 예를 들어, MNIST 아라비아 숫자 데이터 (그림) 를 학습한 VAE, C-VAE 모델이 있다고 할 때, 기존의 VAE가 **아라비아 숫자를 하나 생성해 주는** 역할을 수행했다면, C-VAE 모델은 **숫자 5를 하나 생성해 주는** 역할을 수행할 수 있다.
* 모델 구조 **(추후 작성)**
  * 전체 모델 ```cvae_model```
  * 인코더 모델 ```cvae_encoder_model```
  * 디코더 모델 ```cvae_decoder_model```

## 실행 순서
```
python train.py
python test.py
```

## 성능지표 결과
* 성능 측정지표 : 정성 평가로 진행
  * ```test.py``` 파일 실행 시, 디코더 모델 (```cvae_decoder_model```) 에서 입력 Class를 받은 후, 모델이 출력하는 이미지 ```test_output.png``` 파일을 확인할 수 있음

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|GAI-P1-master|||240302|240310|마스터 브랜치|
|GAI-P1-1|```done```|```feat```|240302|240302|```mnist_train.csv``` 데이터를 C-VAE 모델로 학습|
|GAI-P1-2|```done```|```feat```|240302|240302|C-VAE 모델 테스트|
|GAI-P1-3||```fix```|||C-VAE 모델 성능 향상 시도|