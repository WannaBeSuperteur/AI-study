# Generative AI 프로젝트 2. Conditional VAE 를 이용한 사람 얼굴 생성하기
* Dataset:
  * [Person Face Dataset (thispersondoesnotexist)](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist)
  * [Face Dataset Of People That Don't Exist](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist)
  * 모두 https://thispersondoesnotexist.com/ 로부터 생성된 데이터
* 수행 기간: 2024.03.11 ~ 2024.03.24 (14일)

## 파일 및 코드 설명
* 학습 데이터셋 구조 (다운받은 데이터를 아래와 같이 배치), **전체 16,873 장**

```
Project 002. Conditional VAE 를 이용한 사람 얼굴 생성하기
- thispersondoesnotexist.10k
  - Person Face Dataset (thispersondoesnotexist) 10,000 장 (original)
- ThisPersonDoesNotExist
  - Female
    - Face Dataset Of People That Don't Exist 3,013 장 (original)
  - Male
    - Face Dataset Of People That Don't Exist 3,860 장 (original)
```

* ```resize_and_sort_data.py``` : 다운받은 데이터셋의 이미지를 리사이징, 명백히 배경에 해당하는 일부분을 cropping 및 재배치
  * 필요 파일 : 2개의 데이터셋으로부터 다운받은 이미지 (위 학습 데이터셋 구조 참고)
  * 출력 파일 : 다음과 같은 디렉토리 구조로 재배치된 resized 이미지

```
Project 002. Conditional VAE 를 이용한 사람 얼굴 생성하기
- resized_images
  - first_dataset
    - Person Face Dataset (thispersondoesnotexist) 10,000 장 (resized)
  - second_dataset_male
    - Face Dataset Of People That Don't Exist 3,013 장 (resized)
  - second_dataset_female
    - Face Dataset Of People That Don't Exist 3,860 장 (resized)
```

* ```train_model_for_conditions_male_or_female.py``` : C-VAE의 condition을 위한 성별 classify 모델 학습
  * 성별 조건
    * **Face Dataset Of People That Don't Exist** 로부터의 데이터셋을 이용하여 **성별 예측 분류 모델 학습**
    * 해당 조건을 C-VAE 모델에 추가로 입력 시 cell의 개수는 2개 (각각 ```male```일 확률, ```female```일 확률, activation = softmax)
    * 출력 모델 : ```classify_male_or_female```
* ```train_model_for_conditions_hair_color.py``` : C-VAE의 condition을 위한 머리 색 regression 모델 학습
  * 머리 색 조건
    * **Face Dataset Of People That Don't Exist** 로부터의 데이터셋 중 **남녀 각각 최초 1000장 (이름순)** 의 사진을 학습 및 validation
    * 머리 색이 진할수록 1에 가깝고, 밝을수록 0에 가까운 값을 출력하는 **regression 모델 학습**
    * 필요 파일 :
      * ```regression_hair_color_info_male.csv``` (남성 최초 1000장 이름순, 머리 색을 나타내는 0-1 값 정보)
      * ```regression_hair_color_info_female.csv``` (여성 최초 1000장 이름순, 머리 색을 나타내는 0-1 값 정보)
    * 출력 모델 : ```regression_hair_color```
* ```train_model_for_conditions_mouth.py``` : C-VAE의 condition을 위한 입을 벌린 정도 regression 모델 학습
  * 입을 벌린 정도
    * **머리 색 조건** 에 대한 데이터셋과 동일한 데이터셋을 학습 및 validation
    * 입을 벌린 정도가 클수록 1에 가깝고, 작을수록 (입을 다물었을수록) 0에 가까운 값을 출력하는 **regression 모델 학습**
    * 필요 파일 :
      * ```regression_mouth_info_male.csv``` (남성 최초 1000장 이름순, 입을 벌린 정도를 나타내는 0-1 값 정보)
      * ```regression_mouth_info_female.csv``` (여성 최초 1000장 이름순, 입을 벌린 정도를 나타내는 0-1 값 정보)
    * 출력 모델 : ```regression_mouth```
* ```train_model_for_conditions_eyes.py``` : C-VAE의 condition을 위한 눈을 뜬 정도 regression 모델 학습
  * 눈을 뜬 정도
    * **머리 색 조건** 에 대한 데이터셋과 동일한 데이터셋을 학습 및 validation
    * 눈을 뜬 정도가 클수록 1에 가깝고, 눈을 감았을수록 0에 가까운 값을 출력하는 **regression 모델 학습**
    * 필요 파일 :
      * ```regression_eyes_info_male.csv``` (남성 최초 1000장 이름순, 눈을 뜬 정도를 나타내는 0-1 값 정보)
      * ```regression_eyes_info_female.csv``` (여성 최초 1000장 이름순, 눈을 뜬 정도를 나타내는 0-1 값 정보)
    * 출력 모델 : ```regression_eyes```
  * 각각의 모델은 간단한 구조의 Convolutional Neural Network 를 이용
  * 각각의 모델 학습 시, **남성 최초 901-1000번째, 여성 최초 901-1000번째 이름순** 데이터를 validation 데이터로 이용
* ```train_cvae_model.py``` : 학습 과정 전체 진행
  * 필요 파일 : resize 및 재배치된 전체 학습 데이터셋 (위 ```resizd_images``` 및 그 내부 디렉토리 구조 참고)
  * 출력 모델 : ```cvae_model``` (Conditional VAE 모델)
* ```test_cvae_model.py``` : 이미지의 condition 이 주어지면 해당 condition 을 이용하여 이미지 생성
  * 필요 모델 : ```cvae_decoder_model``` (디코더), C-VAE의 condition을 위한 분류/회귀 모델들
  * 출력 파일 : ```test_output.png```

## 데이터 전처리 및 생성 과정
* 이미지 리사이징 후 데이터셋 재배치 진행

## 머신러닝 모델 설명
* Conditional VAE 모델을 이용
  * VAE 참고 : [Variational Auto-Encoder](https://github.com/WannaBeSuperteur/AI-study/blob/main/Generative%20AI/Basics_Variational%20Auto%20Encoder.md)
  * Conditional VAE (C-VAE) 는 VAE를 통해 생성하려는 데이터에 **제약 조건** 을 추가한 형태의 모델
  * 예를 들어, MNIST 아라비아 숫자 데이터 (그림) 를 학습한 VAE, C-VAE 모델이 있다고 할 때, 기존의 VAE가 **아라비아 숫자를 하나 생성해 주는** 역할을 수행했다면, C-VAE 모델은 **숫자 5를 하나 생성해 주는** 역할을 수행할 수 있다.
* Conditional VAE 모델의 제약 조건
  * 성별 조건 (```classify_male_or_female``` 모델의 출력 결과를 이용)
  * 머리 색 조건 (```regression_hair_color``` 모델의 출력 결과를 이용)
  * 입을 벌린 정도 (```regression_mouth``` 모델의 출력 결과를 이용)
  * 눈을 뜬 정도 (```regression_eyes``` 모델의 출력 결과를 이용)
  * 이상 4개 모델의 **학습 및 validation에 사용한 데이터** 에 대해서도 마찬가지로 **해당 모델의 출력 결과를 이용**
* Conditional VAE 모델의 Loss function
  * Total loss = ```MSE Loss``` + ```KL Divergence``` + ```??? Loss```
    * 단, ```??? Loss``` 는 2nd epoch 부터 적용
  * MSE Loss : 모든 pixel 에 대한 mean squared error의 합
  * **??? Loss** : 디코더 모델 ```cvae_decoder_model``` 의 최종 출력을 보다 선명하게 만들기 위해서 추가하는 Loss 값
* 모델 구조 (그림 추후 추가 예정)
  * 전체 모델 ```cvae_model```
  * 인코더 모델 ```cvae_encoder_model```
  * 디코더 모델 ```cvae_decoder_model```

## 실행 순서
```
python resize_and_sort_data.py
python train_model_for_condition_male_or_female.py
python train_model_for_condition_hair_color.py
python train_model_for_condition_mouth.py
python train_model_for_condition_eyes.py
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
|GAI-P2-1|```done```|```feat```|240312|240312|data 재배치 및 resizing, 배경 부분 cropping 실시|
|GAI-P2-2|```ing```|```feat```|240313||```성별 조건``` condition을 위한 ```classify_male_or_female``` 모델 학습|
|GAI-P2-3||```feat```|||```머리 색 조건``` condition을 위한 ```regression_hair_color``` 모델 학습|
|GAI-P2-4||```feat```|||```입을 벌린 정도``` condition을 위한 ```regression_mouth``` 모델 학습|
|GAI-P2-5||```feat```|||```눈을 뜬 정도``` condition을 위한 ```regression_eyes``` 모델 학습|
|GAI-P2-6||```feat```|||C-VAE 모델 학습|
|GAI-P2-7||```feat```|||C-VAE 모델 테스트 (사람 얼굴 이미지 생성)|
|GAI-P2-8||```feat```|||C-VAE 모델 성능 개선 (특별한 Loss 추가, 새로운 condition 추가, 모델 구조 변경 등)|