## 목차

* [1. 개요](#1-개요)
* [2. 실험 상세](#2-실험-상세)
  * [2-1. 이미지 회전 각도 판단](#2-1-이미지-회전-각도-판단)
  * [2-2. 글자 직사각형 영역 탐지](#2-2-글자-직사각형-영역-탐지)
  * [2-3. 텍스트 인식 (글자 분류)](#2-3-텍스트-인식-글자-분류)
* [3. 실험 결과](#3-실험-결과)

## 1. 개요

* [OCR의 기본 프로세스](OCR_Basics.md#2-ocr의-동작-원리) 에 따른 간단한 OCR 모델을 만들어서 테스트한다.
* **OCR 실무 경험 이전** 의 실험으로, 향후 **OCR 실무 경험 이후** 에 본 문서의 내용보다 OCR에 대한 이해가 얼마나 향상되었는지 평가한다.
* [OCR 실험 상세 코드](Special%20-%20OCR%20Experiment)

## 2. 실험 상세

* OCR 의 기본 프로세스의 각 과정에 다음과 같이 알고리즘 및 딥러닝 모델을 적용한다.

![image](images/OCR_Basic_1.PNG)

| OCR 프로세스         | 알고리즘 / 딥러닝 모델                                                                                          | 데이터셋<br>(학습 또는 테스트 목적)                                                                                                                                                                                                 |
|------------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 이미지 획득           | 단순 색 변환 (원본 색 → Black or White)                                                                        |                                                                                                                                                                                                                        |
| 이미지 전처리 및 문자 세분화 | - 이미지 회전 각도 판단 **(CNN 모델)** 및 정방향으로 회전 처리<br>- 회전 처리된 이미지에 대해 글자를 나타내는 직사각형 영역 탐지 **(AI가 아닌 단순 알고리즘)** | [Scanned Images Dataset for OCR and VLM finetuning (from Kaggle)](https://www.kaggle.com/datasets/suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning) > ```Letter``` ```Memo``` ```Report``` 의 회전되지 않은 이미지 중 80% |
| 텍스트 인식           | - 탐지된 직사각형 영역에 있는 글자를 분류 **(CNN 모델)**                                                                  | [Standard OCR Dataset (from Kaggle)](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset) > ```data``` directory                                                                                            |
| 최종 결과 도출         | - OCR Model 를 이용하여 탐지된 결과를 최종 합성                                                                       |                                                                                                                                                                                                                        |
| (최종 성능 테스트)      |                                                                                                        | [Scanned Images Dataset for OCR and VLM finetuning (from Kaggle)](https://www.kaggle.com/datasets/suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning) > ```Letter``` ```Memo``` ```Report``` 의 회전되지 않은 이미지 중 20% |

* 데이터셋 선정 기준
  * hand-writting 이 아닌 컴퓨터로 쓴 글자 형태의 데이터셋
  * 자유 사용 가능 라이선스
  * 회전 각도가 labeling 되어 있거나, 이미지의 대부분이 정방향인 문서 촬영 사진 데이터셋

* 데이터셋 변형

| 데이터셋                                                                                                                                                        | 변형 방법                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| [Scanned Images Dataset for OCR and VLM finetuning (from Kaggle)](https://www.kaggle.com/datasets/suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning) | 랜덤 각도 회전 (회전 각도 라벨링)         |
| [Standard OCR Dataset (from Kaggle)](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset)                                                        | 배경 흰색 & 글자 검은색 (완전 흑백) 으로 변환 |

### 2-1. 이미지 회전 각도 판단

* 모델 정보
  * Document 이미지의 회전 각도 예측 (**-15도 ~ +15도** 범위를 **0.0 ~ 1.0** 으로 linear 하게 변환)
  * **Pre-trained ResNet18** 기반
* 모델 성능

| MSE Loss (= MSE Error) | MAE Error            |
|------------------------|----------------------|
| 0.0013                 | 0.0275 (0.83 degree) |

### 2-2. 글자 직사각형 영역 탐지

* 흰색 이미지에서 검은색 글자 영역 추출
  * 해당 영역의 가로/세로 길이 및 비율 조건 만족 시 글자로 판정
* BFS 알고리즘 기반
* [상세 구현 코드](Special%20-%20OCR%20Experiment/run_extract_letters.py)

### 2-3. 텍스트 인식 (글자 분류)

* 모델 정보
  * 텍스트를 ```0``` ```1``` ```2``` ... ```Z``` 의 36가지 중 하나로 분류
  * **Pre-trained ResNet18** 기반
* 모델 성능

| Image Size<br>(after resized) | MSE Loss<br>(= MSE Error) | Accuracy               |
|-------------------------------|---------------------------|------------------------|
| 256 x 256                     | 2.6608                    | 0.9702 (= 97.02 %)     |
| 128 x 128                     | **2.6573**                | **0.9742 (= 97.42 %)** |
| 64 x 64                       | 2.6806                    | 0.9494 (= 94.94 %)     |
| 32 x 32                       | 2.6809                    | 0.9494 (= 94.94 %)     |

* [Confusion Matrix (with image size 128 x 128)](Special%20-%20OCR%20Experiment/cf_matrix_letter_classify_model.csv)

## 3. 실험 결과

* 결론
  * 글자 인식 정확도 **매우 낮음 (거의 인식되지 않음)** 또는 **배열 순서 오류**
  * 향후 OCR 실무 수행 후, 재 학습 및 성능 향상 시도

* 실험 대상 이미지

![image](Special%20-%20OCR%20Experiment/test_black_white.png)

* 실험 결과

```
QLV6PSNX90C0108U6C3KD2VX6209X0002222939F2022J2X60C1NU0EICN221991MCC811HGCL11C0QPS80L0Y0C00R0N88TCP1GQH1I1SL0VPP1C1061000N00LWP1RCC0S01G6Q16PLCHSCPL1011Y81C502C0Y1RC00008P1G61LHHQCLL0P10508L100Y8RC0800CQP111HLL11MK50D1FXL0L1ALLLC
5C06TL00C0S05111C01C0AB0N9116L5301SJ0C01L11LHC11021NC61N00L001S00L00C0I0Q6LE05090IPU0Y01E6LX00WL6S8QC5086000L01103905UCCQ151N6199176Y1FL180NQC01NI21990C60S6LNLA1R0C1QNC0B1CCNJ111M0XB1N06Y001LC0NLR00Q51110CS00N16KX8NNI1GY0XWK100C
0M016S6L5S1N0111Q0N0C9NL11I299L01LAR1C0E1060CX0LX050P1IL80D0N1506X0C6L0N1L19001910L0S11116060L1I0H11LLSYQC0P8C010LLNLL1Y1Z0I51106P0610DL10I1N911Q0L19906N01N90C11616CL10XPXLSC061C00YC10N5S00INRTX0FCENGU9E8II0C0NN8CTI0408T286CUCT8
DAININGPR0C0112616LLCL1L2L601S00000C000060LWV01Y0NN01011L11C50519C0PC9CP01N9E5HLC05LP2UX01N10MH66L1020115P1CLL6CLS6L0N0CL1C0002L10H6LL5WY211111LL10C11CC2551096LCPC0L01501N0X11LN511611C00XC0116NN081661L0660YL08N6P0C0L1L00XP111016
L019106L62Y6L0L0P0XC010C00Z151P1219XL1901106L1PY601L01C0N0L112L13P0510S1C6L0192X4W8EWWUFU941444F
```
