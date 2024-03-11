# Generative AI 프로젝트 2. Conditional VAE 를 이용한 사람 얼굴 생성하기
* Dataset: [MNIST DataSet](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)
  * 위 주소에서 다운로드 받은 ```mnist_train.csv``` 파일이 학습 데이터이다.
* 수행 기간: 2024.03.11 ~ 2024.03.24 (14일)

## 파일 및 코드 설명
* 학습 데이터셋 구조

```
Project 002. Conditional VAE 를 이용한 사람 얼굴 생성하기
- archive
  - fake
    - 700 images
  - real
    - 589 images
  - data.csv
```

* ```augmentation.py``` : 이미지 데이터 augmentation 용
  * 필요 파일 : ```archive``` 폴더 내부의 학습 데이터셋
  * 출력 파일 : 다음과 같이 새로운 학습 데이터셋 구조를 갖는다. (단, N은 augmentation 시 경우의 수로, augmentation에 의해 이미지 개수가 증가되는 배수)

```
Project 002. Conditional VAE 를 이용한 사람 얼굴 생성하기
- augmentation
  - N x (700 + 589) images
```

* ```train_models_for_conditions.py``` : C-VAE의 condition을 위한 분류/회귀 모델들을 학습
  * 어떤 조건 (예: 성별, 표정, 움직임) 으로 할지는 추후 고려
  * 경우에 따라 분류/회귀 모델이 아닌 일반 알고리즘을 이용할 수도 있음
* ```train_cvae_model.py``` : 학습 과정 전체 진행
  * 필요 파일 : ```augmentation``` 폴더 내부의 학습 데이터셋
  * 출력 모델 : ```cvae_model``` (Conditional VAE 모델)
* ```test_cvae_model.py``` : 이미지의 condition 이 주어지면 해당 condition 을 이용하여 이미지 생성
  * 필요 모델 : ```cvae_decoder_model``` (디코더), C-VAE의 condition을 위한 분류/회귀 모델들
  * 출력 파일 : ```test_output.png```

## 데이터 전처리 및 생성 과정
* Data Augmentation을 먼저 진행 (채도, crop, 회전 위주)
  * 세부 사항 추가 예정
* Augmentation 된 이미지를 **가장자리 일부분을 잘라내어** ```augmentation``` 폴더에 저장
* 학습 전에 이미지를 일정한 크기로 resize

## 머신러닝 모델 설명
* Conditional VAE 모델을 이용
  * VAE 참고 : [Variational Auto-Encoder](https://github.com/WannaBeSuperteur/AI-study/blob/main/Generative%20AI/Basics_Variational%20Auto%20Encoder.md)
  * Conditional VAE (C-VAE) 는 VAE를 통해 생성하려는 데이터에 **제약 조건** 을 추가한 형태의 모델
  * 예를 들어, MNIST 아라비아 숫자 데이터 (그림) 를 학습한 VAE, C-VAE 모델이 있다고 할 때, 기존의 VAE가 **아라비아 숫자를 하나 생성해 주는** 역할을 수행했다면, C-VAE 모델은 **숫자 5를 하나 생성해 주는** 역할을 수행할 수 있다.
* Conditional VAE 모델의 제약 조건
  * 추가 예정
* Conditional VAE 모델의 Loss function
  * Total loss = ```MSE Loss``` + ```KL Divergence``` + ```??? Loss```
    * 단, ```??? Loss``` 는 2nd epoch 부터 적용
  * MSE Loss : 모든 pixel 에 대한 mean squared error의 합
  * **??? Loss** : 디코더 모델 ```cvae_decoder_model``` 의 최종 출력을 보다 선명하게 만들기 위해서 추가하는 Loss 값
* 모델 구조 (아래 그림 참고)
  * 전체 모델 ```cvae_model```
  * 인코더 모델 ```cvae_encoder_model```
  * 디코더 모델 ```cvae_decoder_model```

## 실행 순서
```
python augmentation.py
python train_models_for_conditions.py
python train.py
python test.py
```

## 성능지표 결과
* 성능 측정지표 : 정성 평가로 진행
  * ```test.py``` 파일 실행 시, 디코더 모델 (```cvae_decoder_model```) 에 condition을 입력 받은 후, 모델이 출력하는 이미지 ```test_output.png``` 파일을 확인할 수 있음

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|GAI-P2-master|||240311|240324|마스터 브랜치|
|GAI-P2-1||```feat```|||data augmentation 실시|
|GAI-P2-2||```feat```|||??? condition을 위한 ??? 모델 학습|
|GAI-P2-3||```feat```|||??? condition을 위한 ??? 모델 학습|
|GAI-P2-4||```feat```|||??? condition을 위한 ??? 모델 학습|
|GAI-P2-5||```feat```|||C-VAE 모델 학습|
|GAI-P2-6||```feat```|||C-VAE 모델 테스트 (사람 얼굴 이미지 생성)|