# Generative AI 프로젝트 1. MNIST 숫자 생성 AI with C-VAE
* Dataset: [MNIST DataSet](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data)
  * 위 주소에서 다운로드 받은 ```mnist_train.csv``` 파일이 학습 데이터이다.
* 수행 기간: 2024.03.02 ~ 2024.03.07 (6일)

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
* Conditional VAE 모델의 제약 조건
  * 숫자 값의 class (```0```, ```1```, ..., ```9```) 를 나타내는 one-hot vector **(10)**
  * **white 성분 비율** : 이미지의 전체 픽셀에 대해 white 성분 값 (0~255) 의 평균을 255로 나눈 값
    * 실제로는 해당 값을 ```x -> {(x - (white 성분 비율의 해당 숫자 평균)) / (white 성분 비율의 해당 숫자 표준편차)} * 0.5``` 를 ```-1.0 ~ 1.0``` 범위로 clipping 한 값으로 처리한 값을 사용
    * **해당 숫자** 는 white 성분 비율을 구하려는 이미지가 나타내는 숫자 (0, 1, ..., 9) 를 의미하며, 같은 숫자에 해당하는 이미지만을 대상으로 평균, 표준편차 계산
* Conditional VAE 모델의 Loss function
  * Total loss = ```MSE Loss``` + ```KL Divergence``` + ```Grey Y Loss``` (2nd epoch ~)
  * MSE Loss : 모든 cell (28 x 28 = 784개) 에 대한 mean squared error의 합
  * **Grey Y Loss** : C-VAE 전체 모델 (```cvae_model```) 의 **최종 출력 이미지** 의 모든 각 pixel을 생각하자. 이들 각각의 pixel에 대해, 그 value를 pixel의 해당 성분 값 (0~255) 을 255로 나눈 값 (검은색 -> 0.0, 흰색 -> 1.0) 이라고 하자. 이때 각 pixel에 대한 ```k * (value) * (1.0 - (value))``` 의 총합을 loss로 정의한다.
    * k 는 다른 Loss 와의 밸런스를 위한 파라미터 값
    * 출력 이미지에서 회색으로 표시되는 부분을 최대한 제거하고 글자를 보다 선명하게 하기 위해 추가한 Loss
    * 단, Grey Y Loss 는 **2번째 epoch부터** 기존 loss에 Grey Y Loss를 더하는 방식으로 적용
      * 1번째 epoch 에서는 기존 ```MSE Loss``` + ```KL Divergence``` 의 값으로 Total Loss를 계산
      * 1번째 epoch 부터 Grey Y Loss 를 적용하면 검은색 이미지를 생성하는 것으로 모델이 수렴하는 현상이 종종 발생

![image](./grey_y_loss.PNG)

* 모델 구조 (아래 그림 참고)
  * 전체 모델 ```cvae_model```
  * 인코더 모델 ```cvae_encoder_model```
  * 디코더 모델 ```cvae_decoder_model```

![image](./model_architecture.PNG)

## 추가 기능
* 간단한 CNN 모델을 이용하여 숫자 이미지 분류
  * ```classify_nums.py```
  * 모델 이름 : ```classify_nums_model```
* 사용자가 그림판으로 그린 숫자를 해당 CNN 모델로 분류한 후, 해당 숫자 이미지를 **```cvae_encoder_model``` 에 입력하여 그 출력으로 얻은 것과 동일한 latent vector와 white 성분 비율, 즉 동일한 글씨체** 로 나머지 class 에 해당하는 숫자를 ```cvae_decoder_model``` 을 이용하여 생성
  * 예를 들어 사용자가 ```3``` 이라는 숫자를 그렸으면, 나머지 ```0```, ```1```, ```2```, ```4```, ..., ```9``` 숫자 생성
  * ```generate_other_nums.py```
  * 필요한 모델 : ```classify_nums_model```, ```cvae_encoder_model```, ```cvae_decoder_model```
  * 필요 파일 : ```number.png``` (사용자가 그린 숫자 그림, 28 x 28 grayscale 이미지, **검은색 배경에 흰 글씨** 이어야 함!!)

## 실행 순서
```
python train.py
python test.py
```

**추가 기능 실행 순서**
* 반드시 먼저 ```train.py``` 를 실행한 후 그 다음에 아래를 순서대로 실행
* ```generate_other_nums.py``` 실행 시, 입력 이미지인 ```number.png``` 파일 필요
* ```generate_other_nums.py``` 실행 시, 출력 이미지 위치는 ```test_imgs/generated_L_A_B.png``` 이다.
  * ```L``` 은 encoder에 생성된 latent vector의 번호 (0, 1, 2, ..., 7)
  *  ```A``` 가 0, 1, 2일 때 생성되는 이미지의 white ratio는 각각 0.0, 0.5, 1.0 이고, ```B``` 는 생성되는 이미지가 나타내는 숫자이다.

```
python classify_nums.py
python generate_other_nums.py
```

## 성능지표 결과
* 성능 측정지표 : 정성 평가로 진행
  * ```test.py``` 파일 실행 시, 디코더 모델 (```cvae_decoder_model```) 에서 입력 Class를 받은 후, 모델이 출력하는 이미지 ```test_output.png``` 파일을 확인할 수 있음

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|GAI-P1-master|||240302|240307|마스터 브랜치|
|GAI-P1-1|```done```|```feat```|240302|240302|```mnist_train.csv``` 데이터를 C-VAE 모델로 학습|
|GAI-P1-2|```done```|```feat```|240302|240302|C-VAE 모델 테스트|
|GAI-P1-3|```done```|```fix```|240302|240305|C-VAE 모델 성능 향상 시도|
|GAI-P1-4|```done```|```feat```|240306|240307|CNN으로 숫자 분류하는 모델 구현|
|GAI-P1-5|```done```|```feat```|240307|240307|사용자가 숫자를 그리면 동일한 글씨체로 나머지 숫자 생성 모델 구현|