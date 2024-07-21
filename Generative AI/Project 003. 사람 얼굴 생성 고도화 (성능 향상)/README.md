# GAI-P3. 사람 얼굴 생성 고도화 (성능 향상)
## 프로젝트 개요
* 진행 일정 : **2024.07.20 (토) ~ 08.11 (일), 23일간**
* 프로젝트 목표 : 생성형 AI (GAN, CVAE, NVAE 등) 를 이용하여, 진짜 같은 사람 얼굴을 만드는 것

## 기술적 접근 방법
* 사람 얼굴 생성 모델 구현
  * NVAE (Nouveau VAE) 의 아이디어를 이용
  * [NVAE official paper](https://arxiv.org/pdf/2007.03898)
* 학습 데이터 결정 모델
  * 학습 데이터 중 성별이 labeling 된 이미지 (Female 3,860 장, Male 3,013 장) 를 학습하여, '학습 데이터 결정 모델' 생성
  * 학습 데이터 결정 모델을 이용하여, 여성일 확률이 99.99% 이상이면서 원래 포함된 디렉토리가 'male' 디렉토리가 아닌 이미지만 학습 데이터셋으로 사용 
* 입력값 결정 모델
  * 학습 데이터셋의 이미지에서 **CVAE 입력값**을 나타내는 연속된 숫자 값을 CVAE의 입력으로 하기 위해, 그 입력값을 결정하는 모델
    * CVAE 입력값 (5종) : **눈을 뜬 정도, 입을 벌린 정도, 헤어 컬러, 배경 색, 고개를 돌린 유형**
  * 학습 데이터 중 다음의 2,000 장에 대해 해당 입력값을 수기로 labeling 하여, 해당 입력값을 다른 이미지에 대해서 결정할 수 있는 Regression Model 학습
    * ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/resized/ThisPersonDoesNotExist/Male``` 에 저장된 남성 이미지 중 파일 이름순 최초 1,000 장
    * ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/resized/ThisPersonDoesNotExist/Female``` 에 저장된 여성 이미지 중 파일 이름순 최초 1,000 장
  * 해당 Regression 모델의 예측값으로, 그 2,000 장을 포함한 전체 학습 데이터에 대해 해당 입력값을 결정
    * 모든 데이터에 대해 모델 예측값을 사용한다는 일관성을 위해, 해당 모델의 학습 데이터도 입력값 결정 대상에 포함
* 사람 얼굴의 어색한 부분 수정
  * (A) 원본 데이터와 (B) NVAE 응용 모델을 이용하여 생성한 이미지 간 구분하는 Classification model 개발
  * 해당 모델을 이용하여 NVAE 응용 모델로 생성한 이미지를 리터칭
* 그림 (TBU)

### 모델 파일 정보
* 학습 데이터 결정 모델
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/models/decide_train_data``` 
  * 전체 이미지에 대한 예측 결과 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/models/data/all_output_decide_train_data.csv``` 
  * 학습 데이터 결정 모델의 학습을 위한 데이터
    * 아래 "가공 데이터셋" 중 **R-D2-F, R-D2-M (성별이 명시된 이미지 총 6,873 장)**
* 입력값 결정 모델 및 관련 데이터
  * 모델 주소의 처음 부분인 ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/models/``` 는 생략

| 입력값        | 입력값 설명<br>(입력값 결정 모델 output / CVAE input)          | 입력값 결정 모델 및 코드 | 입력값 결정 모델을 위한<br>학습 데이터 output<br>(수기로 라벨링) | 입력값 결정 모델에 의한<br>모든 이미지 output   | 
|------------|----------------------------------------------------|------------------------|-------------------------------------------|----------------------------------|
| eyes       | **눈을 뜬 정도 👀**<br>- 0~1                            | ```input_eyes```<br>```model_input_eyes.py```      | ```data/train_output_eyes.csv```          | ```data/all_output_eyes.csv```   |
| hair_color | **헤어 컬러 👩‍🦰**<br>- 0~1, 진할수록 1에 가까움              | ```input_hair_color```<br>```model_input_hair_color.py``` | ```data/train_output_hair_color.csv```    | ```data/all_output_hair_color.csv``` |
| mouth      | **입을 벌린 정도 👄**<br>- 0~1                           | ```input_mouth```<br>```model_input_mouth.py```     | ```data/train_output_mouth.csv```         | ```data/all_output_mouth.csv```  |
| background | **배경의 밝기 🏞**<br>- 0~1, 밝을수록 1에 가까움                | ```input_background```<br>```model_input_background.py``` | ```data/train_output_background.csv```    | ```data/all_output_background.csv``` |
| head       | **고개 돌림 유형 🐴**<br>- 0 (왼쪽), 1 (오른쪽), 0.5 (돌리지 않음) | ```input_head```<br>```model_input_head.py```      | ```data/train_output_head.csv```          | ```data/all_output_head.csv```   |

## 사용 데이터셋
### 원본 데이터셋
* (D1) [Person Face Dataset (thispersondoesnotexist)](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist)
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/original/thispersondoesnotexist.10k``` (10,000 장의 이미지)
* (D2) [Face Dataset Of People That Don't Exist](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist)
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/original/ThisPersonDoesNotExist/Female``` (3,860 장의 이미지)
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/original/ThisPersonDoesNotExist/Male``` (3,013 장의 이미지)
* 모두 https://thispersondoesnotexist.com/ 로부터 생성된 데이터

|데이터셋|전체 규모| 남성 이미지 | 여성 이미지 |
|---|---|--------|--------|
|D1|10,000| -      | -      |
|D2|6,873| 3,013  | 3,860  |
|total|16,873| -      | -      |

### 가공 데이터셋
* 원본 데이터셋에 대해 crop + resize + augmentation (brightness) 적용
* 데이터 경로에서 처음의 ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)``` 부분은 생략

| 데이터셋   | 데이터 경로                           | crop + resize | augmentation | 데이터 규모 | 원본 데이터 |
|--------|----------------------------------|---------------|--------------|--------|--------|
| R-D1   | ```dataset/resized/10k-images``` | O             | X            | 10,000 | D1     |
| R-D2-F | ```dataset/resized/female```             | O             | X            | 3,860  | D2     |
| R-D2-M | ```dataset/resized/male```               | O             | X            | 3,013  | D2     |
| A-D1   | ```dataset/augmented/10k-images```       | O             | O            | 10,000 | D1     |
| A-D2-F | ```dataset/augmented/female```           | O             | O            | 3,860  | D2     |
| A-D2-M | ```dataset/augmented/male```             | O             | O            | 3,013  | D2     |
| total  |                                  |               |              | 33,746 |        |

* 위 표의 데이터셋에서, 다음 조건을 모두 만족시키는 데이터의 집합을 **최종 데이터셋**으로 사용
  * **학습 데이터 결정 모델**에 의해 **여성일 확률이 99.99% 이상**으로 판정된 이미지
  * 데이터 경로가 ```dataset/resized/male``` 또는 ```dataset/augmented/male``` 에 속하지 않는 이미지

### 최종 데이터셋
NVAE-idea-based CVAE 모델을 학습시키기 위한 최종 데이터셋
* 데이터셋 규모 : **13,064 장 (가공 데이터셋의 38.7%)**
  * 가공 데이터셋 전체의 여성 이미지를 **16,720 장** (10k-images 의 50%를 여성으로 가정) 시, 이들 중 **78.1%**
* 밝기 augmentation 적용 여부만 다른 이미지를 동일 이미지로 간주 시, **6,600 장** 정도로 추정

최종 데이터셋 경로
* 데이터셋 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/final```
* 데이터셋 저장 파일 전체 경로 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/final/{aug_or_res}_{dir_name}_{file_name}.jpg```
  * ```aug_or_res``` : 'augmented' 또는 'resized'
  * ```dir_name``` : 가공 데이터셋 경로 기준의 디렉토리 이름 ('10k-images', 'female', 'male' 중 하나)
  * ```file_name``` : 디렉토리 이름을 제외한 이미지 파일 이름

## 코드 설명 및 실행 순서
설명 및 그림 (TBU)

## 테스트 환경 및 방법
* 테스트 환경 기본 사항
  * Python 3.8
  * Windows 10
  * Quadro M6000 GPU
* Python 라이브러리 정보
  * TensorFlow 등 라이브러리 버전 목록 (TBU)
* 테스트 방법
  * TBU

## 상세 일정
* status : 🔥 (개발 진행중), ✔ (개발 완료)

|개발 사항| 예상 일정              |status|
|---|--------------------|---|
|데이터셋 수집| 07.20 (1d)         |✔|
|데이터 증강 (augmentation)| 07.20 (1d)         |✔|
|학습 데이터 결정 모델 학습| 07.20 (1d)         |✔|
|학습 데이터 결정| 07.20 (1d)         |✔|
|입력값 결정 모델 학습 (학습 데이터는 기존 GAI-P2 에서 만든 데이터를 이용)| 07.20 - 07.21 (2d) |✔|
|사람 얼굴 생성 모델 구현 : NVAE + CVAE 아이디어를 이용하여 접근| 07.21 - 07.26 (6d) |🔥|
|어색한 부분 수정 구현 : 분류 모델 개발| 07.27 - 07.28 (2d) ||
|어색한 부분 수정 구현 : 이미지 리터칭 개발| 07.29 - 08.02 (5d) ||
|모델 전체 테스트| 08.03 - 08.04 (2d) ||
|성능 추가 향상 시도| 08.05 - 08.11 (7d) ||

## 브랜치 정보
* status : 🔥 (개발 진행중), ✔ (개발 완료)
* type : ✨ (feature), 🛠 (bug fix), ⚡ (improve performance)

| branch        |status|type| start    | end      | description            |
|---------------|---|---|----------|----------|------------------------|
| GAI-P3-master |🔥|| 24.07.20 | 24.08.11 | 마스터 브랜치                |
| GAI-P3-1      |✔|✨| 24.07.20 | 24.07.20 | 데이터 수집 및 증강            |
| GAI-P3-2      |✔|✨| 24.07.20 | 24.07.20 | 학습 데이터 결정 모델 생성        |
| GAI-P3-3      |✔|✨| 24.07.20 | 24.07.21 | 입력값 결정 모델 생성           |
| GAI-P3-4      |✔|🛠| 24.07.21 | 24.07.21 | 데이터셋 경로 재설정            |
| GAI-P3-5      ||✨| 24.07.21 |          | 사람 얼굴 생성 모델 구현         |
| GAI-P3-6      ||✨|          |          | 어색한 부분 수정 (분류 모델 개발)   |
| GAI-P3-7      ||✨|          |          | 어색한 부분 수정 (이미지 리터칭 개발) |


## 실험 결과 및 로그

| 작성일      | 로그 내용               | 로그 파일 주소                                       |
|----------|---------------------|------------------------------------------------|
| 24.07.20 | 학습 데이터 결정 모델의 학습 로그 | ```logs/trainlog_train_data_decide_model.md``` |
| 24.07.21 | 입력값 결정 모델의 학습 로그    | ```logs/trainlog_input_decide_model.md```      |