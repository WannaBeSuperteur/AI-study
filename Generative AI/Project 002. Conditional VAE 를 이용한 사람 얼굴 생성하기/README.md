# Generative AI 프로젝트 2. Conditional VAE 를 이용한 사람 얼굴 생성하기
* Dataset:
  * [Person Face Dataset (thispersondoesnotexist)](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist)
  * [Face Dataset Of People That Don't Exist](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist)
  * 모두 https://thispersondoesnotexist.com/ 로부터 생성된 데이터
* 수행 기간: 2024.03.11 ~ 03.17 + 2024.04.01 ~ 04.21 (28일)

## 파일 및 코드 설명
### 데이터셋 구조 관련
* 학습 데이터셋 구조 (다운받은 데이터를 아래와 같이 배치), **전체 16,873 장**

```
Project 002. Conditional VAE 를 이용한 사람 얼굴 생성하기
- thispersondoesnotexist.10k
  - Person Face Dataset (thispersondoesnotexist) 10,000 장 (original)
- ThisPersonDoesNotExist
  - Female
    - Face Dataset Of People That Don't Exist 3,860 장 (original)
  - Male
    - Face Dataset Of People That Don't Exist 3,013 장 (original)
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

### additional info 학습 관련
* additional info 값 출력 모델
  * **일부 이미지에 대해 info 값을 인위적으로 지정하여, CNN 딥러닝 모델이 해당 값을 output 으로 학습한 후, 모든 이미지에 대해서 해당 모델이 출력한 값의 prediction을 이용**
  * ```classify_male_or_female``` 성별 분류 모델 : **Face Dataset Of People That Don't Exist** 로부터의 데이터셋을 이용하여 **성별 예측 분류 모델 학습**
  * 나머지 모든 회귀 모델 (6개) : **Face Dataset Of People That Don't Exist** 로부터의 데이터셋 중 **남녀 각각 최초 1000장 (이름순)** 의 사진을 학습 및 validation
    * 필요 파일 :
      * ```regression_{info_name}_info_male.csv``` (```resized_images/second_dataset_male``` 의 이미지 중 최초 1000장 이름순, 해당 info의 값을 나타내는 0-1 값 정보)
      * ```regression_{info_name}_info_female.csv``` (```resized_images/second_dataset_female``` 의 이미지 중 최초 1000장 이름순, 해당 info의 값을 나타내는 0-1 값 정보)
      * 이상의 2개의 csv 파일은 본 repository에서는 txt 파일로 저장되어 있음
      * ```male_or_female_classify_result_for_all_images.csv``` (모든 이미지에 대한 각 성별일 확률에 대한 정보로, 해당 info condition의 값을 출력하기 위한 모델의 input 값으로 사용)
    * 출력 모델 이름 : ```regression_{info_name}```

* 성별 분류 모델 상세
  * ```train_model_for_conditions_male_or_female.py``` : C-VAE의 condition을 위한 **성별 classify** 모델 학습
    * 해당 조건을 C-VAE 모델에 추가로 입력 시 cell의 개수는 2개 (각각 ```male```일 확률, ```female```일 확률, activation = softmax)
    * 출력 모델 : ```classify_male_or_female```
    * 출력 파일 : ```male_or_female_classify_result_for_all_images.csv```
      * **resized_image** 디렉토리에 있는 **모든 이미지 (학습 데이터 포함)** 에 대한 각 성별일 확률을 저장한 csv 파일
      * 본 repository에는 ```male_or_female_classify_result_for_all_images.txt``` 파일로 저장되어 있으며, csv 로 확장자 변경하여 사용 가능

* 회귀 모델 구성 (총 6개)

|```info_name``` 의 값|회귀 모델 학습 데이터의 output의 의미|
|---|---|
|```hair_color```|**머리 색** 이 진할수록 1에 가깝고, 밝을수록 0에 가까운 값|
|```mouth```|**입을 벌린 정도** 가 클수록 1에 가깝고, 작을수록 (입을 다물었을수록) 0에 가까운 값|
|```eyes```|**눈을 뜬 정도** 가 클수록 1에 가깝고, 눈을 감았을수록 0에 가까운 값|
|```background_mean```|**배경 영역** 의 밝기가 밝을수록 1에 가깝고, 어두울수록 0에 가까운 값|
|```background_std```|**배경 영역** 의 색 (특히 인접한 픽셀/픽셀 그룹 간) 이 일정하지 않을수록 1에 가깝고, 일정할수록 0에 가까운 값|
|```other_person``` **(평균으로 수렴, 실패한 모델)**|**다른 사람이 보일수록** 1에 가깝고, 중심 인물 한 명만 있을수록 0에 가까운 값|

* 회귀 모델에 대한 추가 설명
  * 각각의 모델은 간단한 구조의 Convolutional Neural Network 를 이용
  * 각각의 모델 학습 시, **남성 이미지 100장, 여성 이미지 100장** 의 데이터를 validation 데이터로 이용

### additional info 저장 관련
* ```save_condition_data.py``` : 모든 각 이미지의 condition 값 **(```other_person``` 값 제외)** 을 pandas DataFrame 화
  * 필요 모델 : 6개의 regression 모델
    * ```regression_hair_color```
    * ```regression_mouth```
    * ```regression_eyes```
    * ```regression_background_mean```
    * ```regression_background_std```
    * ```regression_other_person``` **(실패한 모델이므로 미사용, 실제로는 5개만 사용)**
  * 필요 파일 : ```male_or_female_classify_result_for_all_images.csv``` (모든 이미지에 대한 성별 예측 정보)
  * 출력 파일 : ```condition_data.csv```
    * 모든 각 이미지의 condition 값을 pandas DataFrame 의 csv 형식으로 저장한 파일
    * 성별에 대한 예측값은 ```male_or_female_classify_result_for_all_images.csv``` 파일에 저장된 값을 사용
    * **각 모델의 학습 대상이 되는 이미지에 대해서도, 실제 값인 ground truth 값이 아닌 모델에 의해 예측된 condition 값을 저장**
    * 본 repository에서는 ```condition_data.txt``` 파일로 저장되어 있으며, csv 로 확장자 변경하여 사용 가능 

* ```add_face_location_info.py``` : face location info 추가
  * 필요 파일 : resized image 전체 (16,873 장 = first dataset 10,000 장 + second dataset 6,873장)
  * 출력 파일 : ```condition_data.csv``` (기존 5개의 열 외에 face location info 열 추가)

### 모델 학습, 저장 및 테스트
* ```train_cvae_model.py``` : 학습 과정 전체 진행
  * 필요 파일 : resize 및 재배치된 전체 학습 데이터셋 (위 ```resizd_images``` 및 그 내부 디렉토리 구조 참고), ```condition_data.csv``` (모든 각 이미지의 condition 값, face location info 3개 열 포함)
  * 출력 모델 : ```cvae_model``` (Conditional VAE 모델), ```cvae_encoder_model``` (C-VAE의 인코더), ```cvae_decoder_model``` (C-VAE의 디코더)
  * decoder 모델 모의 테스트용 파일 : ```decoder_mock_test.py```

* ```test_cvae_model.py``` : 이미지의 condition 이 주어지면 해당 condition 을 이용하여 이미지 생성
  * 필요 모델 : ```cvae_decoder_model``` (디코더)
  * 출력 파일 : ```test_outputs/test_output_{info}_{N}.png```
    * ```info``` : (남성 ```M``` / 여성 ```F```) + (hair color 5/8/1) + (mouth 0/5/1) + (eyes 5/8/1)
      * 0/5/1 은 해당 값이 각각 0.0, 0.5, 1.0 임을 의미
      * 5/8/1 은 해당 값이 각각 0.5, 0.8, 1.0 임을 의미
    * 예를 들어, ```test_outputs/test_output_F101_{N}.png``` 는 **여성 + 진한 머리 색 (1.0) + 입을 다문 상태 (0.5) + 크게 뜬 눈 (1.0)** 의 이미지를 의미
    * ```N``` 은 동일 ```info``` 값의 이미지가 여러 장 있을 때, 해당 이미지 각각을 나타내는 이미지 번호

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
  * 그 외 2개의 회귀 모델 : ```regression_background_mean```, ```regression_background_std```
  * 이상 7개 모델의 **학습 및 validation에 사용한 데이터** 에 대해서도 마찬가지로 **해당 모델의 출력 결과를 이용**
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
python train_model_for_condition_background_mean.py
python train_model_for_condition_background_std.py
python train_model_for_condition_other_person.py (실패한 모델이므로, 선택)
python save_condition_data.py
python add_face_location_info.py
python train_cvae_model.py
python test_cvae_model.py
```

## 테스트 코드
* ```test_compute_mse_for_all_mean.py```
  * regression output 값 정보를 저장한 csv 파일 (예: ```regression_hair_color_info_male.csv```) 에서, **모든 output 값을 해당 파일의 모든 output 값의 평균값으로 예측** 했을 때의 MSE (mean square error) 값을 출력한다.
  * 파일 이름은 ```file_name``` 변수에 저장되어 있으며, 다른 파일을 테스트하려면 해당 변수의 값을 수정한 후 테스트하면 된다.

## 성능지표 결과
* 성능 측정지표 : 정성 평가로 진행
  * ```test.py``` 파일 실행 시, 디코더 모델 (```cvae_decoder_model```) 에 condition을 입력 받은 후, 모델이 출력하는 이미지 ```test_output.png``` 파일을 확인할 수 있음

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|GAI-P2-master|||240311|240421|마스터 브랜치|
|GAI-P2-1|```done```|```feat```|240312|240312|data 재배치 및 resizing, 배경 부분 cropping 실시|
|GAI-P2-2|```done```|```feat```|240313|240317|```성별 조건``` condition을 위한 ```classify_male_or_female``` 모델 학습|
|GAI-P2-3|```done```|```feat```|240401|240402|```머리 색 조건``` condition을 위한 ```regression_hair_color``` 모델 학습|
|GAI-P2-4|```done```|```feat```|240402|240402|```입을 벌린 정도``` condition을 위한 ```regression_mouth``` 모델 학습|
|GAI-P2-5|```done```|```feat```|240402|240404|```눈을 뜬 정도``` condition을 위한 ```regression_eyes``` 모델 학습|
|GAI-P2-6|```done```|```feat```|240404|240404|condition 데이터를 csv 파일로 저장|
|GAI-P2-7|```done```|```feat```|240406|240407|C-VAE 모델 학습|
|GAI-P2-8|```done```|```feat```|240407|240407|C-VAE 모델 테스트 (사람 얼굴 이미지 생성)|
|GAI-P2-9|```done```|```feat```|240407|240421|C-VAE 모델 성능 개선 (특별한 Loss 추가, 새로운 condition 추가, 모델 구조 변경 등)|
|GAI-P2-10|```done```|```feat```|240411|240420|C-VAE 모델 성능 개선 (on GPU)|
|GAI-P2-11|```done```|```feat```|240419|240420|C-VAE 모델 성능 개선 (새로운 condition 추가)|
