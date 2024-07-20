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
  * 학습 데이터 결정 모델을 이용하여 여성일 확률이 90% 이상인 이미지만 학습 데이터셋으로 사용 
* 입력값 결정 모델
  * 학습 데이터셋의 이미지에서 **눈을 뜬 정도, 입을 벌린 정도, 헤어 컬러, 배경 색** 을 나타내는 연속된 숫자 값을 CVAE의 입력으로 하기 위해, 그 입력값을 결정하는 모델
  * 학습 데이터 중 ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/ThisPersonDoesNotExist/Female``` 에 저장된 여성 이미지 중 이름순 최초 1,000 장에 대해 해당 입력값을 labeling 하여, 해당 입력값을 다른 이미지에 대해서 결정할 수 있는 Regression Model 학습
  * 해당 Regression 모델을 이용하여, 그 1,000 장을 제외한, CVAE 의 나머지 학습 데이터에 대해서도 해당 입력값을 결정
* 사람 얼굴의 어색한 부분 수정
  * (A) 원본 데이터와 (B) NVAE 응용 모델을 이용하여 생성한 이미지 간 구분하는 Classification model 개발
  * 해당 모델을 이용하여 NVAE 응용 모델로 생성한 이미지를 리터칭
* 그림 (TBU)

### 모델 파일 정보
* 학습 데이터 결정 모델
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/models/decide_train_data``` 
  * 전체 이미지에 대한 예측 결과 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/models/all_output_decide_train_data.csv``` 
  * 학습 데이터 결정 모델의 학습을 위한 데이터
    * 아래 "가공 데이터셋" 중 **R-D2-F, R-D2-M (성별이 명시된 이미지 총 6,873 장)**
* 입력값 결정 모델 및 관련 데이터
  * 모델 주소의 처음 부분인 ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/models/``` 는 생략

| 입력값        | 입력값 설명                   | 입력값 결정 모델              | 입력값 결정 모델을 위한, 학습 데이터 output      | 입력값 결정 모델이 판정한, 모든 이미지에 대한 output | 
|------------|--------------------------|------------------------|-----------------------------------|-----------------------------------|
| eyes       | 눈을 뜬 정도, 0~1             | ```input_eyes```       | ```train_output_eyes.csv```       | ```all_output_eyes.csv```         |
| hair_color | 헤어 컬러, 0~1, 진할수록 1에 가까움  | ```input_hair_color``` | ```train_output_hair_color.csv``` | ```all_output_hair_color.csv```   |
| mouth      | 입을 벌린 정도, 0~1            | ```input_mouth```      | ```train_output_mouth.csv```      | ```all_output_mouth.csv```        |
| background | 배경의 밝기, 0~1, 밝을수록 1에 가까움 | ```input_background```     | ```train_output_background.csv``` | ```all_output_background.csv```   |

## 사용 데이터셋
### 원본 데이터셋
* (D1) [Person Face Dataset (thispersondoesnotexist)](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist)
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/thispersondoesnotexist.10k``` (10,000 장의 이미지)
* (D2) [Face Dataset Of People That Don't Exist](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist)
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/ThisPersonDoesNotExist/Female``` (3,860 장의 이미지)
  * 저장 위치 : ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)/dataset/ThisPersonDoesNotExist/Male``` (3,013 장의 이미지)
* 모두 https://thispersondoesnotexist.com/ 로부터 생성된 데이터

|데이터셋|전체 규모| 남성 이미지 | 여성 이미지 |
|---|---|--------|--------|
|D1|10,000| -      | -      |
|D2|6,873| 3,013  | 3,860  |
|total|16,873| -      | -      |

### 가공 데이터셋
* 원본 데이터셋에 대해 crop + resize + augmentation (brightness) 적용
* 데이터 경로에서 처음의 ```Project 003. 사람 얼굴 성능 고도화 (성능 향상)``` 부분은 생략

| 데이터셋   | 데이터 경로                     | crop + resize | augmentation | 데이터 규모 | 원본 데이터 |
|--------|----------------------------|---------------|--------------|--------|--------|
| R-D1   | ```resized/10k-images```   | O             | X            | 10,000 | D1     |
| R-D2-F | ```resized/female```       | O             | X            | 3,860  | D2     |
| R-D2-M | ```resized/male```         | O             | X            | 3,013  | D2     |
| A-D1   | ```augmented/10k-images``` | O             | O            | 10,000 | D1     |
| A-D2-F | ```augmented/female```       | O             | O            | 3,860  | D2     |
| A-D2-M | ```augmented/male```         | O             | O            | 3,013  | D2     |
| total  |                            |               |              | 33,746 |        |

* 위 표의 데이터셋에서, **학습 데이터 결정 모델**에 의해 '여성일 확률이 90% 이상'으로 판정된 이미지만을 데이터셋으로 사용
* 설명 및 그림 (TBU)

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
|입력값 결정 모델 학습 (학습 데이터는 기존 GAI-P2 에서 만든 데이터를 이용)| 07.20 - 07.21 (2d) |🔥|
|사람 얼굴 생성 모델 구현 : NVAE + CVAE 아이디어를 이용하여 접근| 07.21 - 07.26 (6d) ||
|어색한 부분 수정 구현 : 분류 모델 개발| 07.27 - 07.28 (2d) ||
|어색한 부분 수정 구현 : 이미지 리터칭 개발| 07.29 - 08.02 (5d) ||
|모델 전체 테스트| 08.03 - 08.04 (2d) ||
|성능 추가 향상 시도| 08.05 - 08.11 (7d) ||

## 브랜치 정보
* status : 🔥 (개발 진행중), ✔ (개발 완료)
* type : ✨ (feature), 🛠 (bug fix), ⚡ (improve performance)

|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|GAI-P3-master|🔥||24.07.20|24.08.11|마스터 브랜치|
|GAI-P3-1|✔|✨|24.07.20|24.07.20|데이터 수집 및 증강|
|GAI-P3-2|✔|✨|24.07.20|24.07.20|학습 데이터 결정 모델 생성|
|GAI-P3-3|🔥|✨|24.07.20||입력값 결정 모델 생성|
|GAI-P3-4||✨|||사람 얼굴 생성 모델 구현|
|GAI-P3-5||✨|||어색한 부분 수정 (분류 모델 개발)|
|GAI-P3-6||✨|||어색한 부분 수정 (이미지 리터칭 개발)|

## 실험 결과 및 로그

|작성일|로그 내용|로그 파일 주소|
|---|---|---|
|24.07.20|학습 데이터 결정 모델의 학습 로그|```logs/trainlog_train_data_decide_model```|