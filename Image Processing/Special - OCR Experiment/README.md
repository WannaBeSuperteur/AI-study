
## 목차

* [1. 개요](#1-개요)
  * [1-1. 각 단계 별 코드](#1-1-각-단계-별-코드) 
* [2. 디렉토리 구성](#2-디렉토리-구성)
* [3. 데이터셋 구조](#3-데이터셋-구조)
  * [3-1. Scanned Images Dataset for OCR and VLM finetuning](#3-1-scanned-images-dataset-for-ocr-and-vlm-finetuning)
  * [3-2. Standard OCR Dataset](#3-2-standard-ocr-dataset) 
* [4. branch 정보](#4-branch-정보)

## 1. 개요

[OCR 실험](../OCR_Experiment.md) 에 사용된 Python 코드

### 1-1. 각 단계 별 코드

| 단계                      | 코드                             |
|-------------------------|--------------------------------|
| 이미지 획득 (흑백 변환)          | ```run_acquire_image.py```     |
| 이미지 회전 각도 파악 모델 학습 및 저장 | ```run_train_angle_model.py``` |

## 2. 디렉토리 구성

| 디렉토리                      | 설명                       |
|---------------------------|--------------------------|
| ```classify_letter```     | 각 글자 영역에 있는 글자를 분류       |
| ```get_rectangle```       | 전처리된 이미지에서 각 글자 영역 도출    |
| ```image_preprocessing``` | 이미지 전처리, 회전 각도 파악, 회전 처리 |

## 3. 데이터셋 구조

### 3-1. Scanned Images Dataset for OCR and VLM finetuning

* 원본 데이터셋
  * [Scanned Images Dataset for OCR and VLM finetuning (from Kaggle)](https://www.kaggle.com/datasets/suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning)
* 데이터셋 구조

```
- Special - OCR Experiment/scanned_images_dataset
  - dataset        (original dataset, 100%)
    - Letter       (543 images)
    - Memo         (599 images)
    - Report       (252 images)
  - train          (80%)
    - Letter       (435 images)
    - Memo         (480 images)
    - Report       (202 images)
  - test           (20%)
    - Letter       (108 images)
    - Memo         (119 images)
    - Report       (50 images)
  - train_rotated  (80%, random angle rotated images)
    - Letter       (435 images)
    - Memo         (480 images)
    - Report       (202 images)
  - test_rotated   (20%, random angle rotated images)
    - Letter       (108 images)
    - Memo         (119 images)
    - Report       (50 images)
```

* 참고 사항
  * 정방향 (회전되지 않은) 이 아닌 이미지, 저화질 이미지 등은 AI 모델 없이 수작업으로 삭제 처리
  * 남아 있는 정방향 이미지만을 학습/테스트 데이터로 사용
  * 데이터셋 train/test 분리 & 회전 처리 : ```split_and_rotate_dataset.py```

### 3-2. Standard OCR Dataset

* 원본 데이터셋
  * [Standard OCR Dataset (from Kaggle)](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset) 
* 데이터셋 구조

```
- Special - OCR Experiment/standard_ocr_dataset
  - data
    - training_data
      - 0            (573 images)
      - 1            (573 images)
      - 2            (573 images)
      - ...
    - testing_data
      - 0            (28 images)
      - 1            (28 images)
      - 2            (28 images)
      - ...
    - training_data_modified
      - 0            (573 images)
      - 1            (573 images)
      - 2            (573 images)
      - ...
    - testing_data_modified
      - 0            (28 images)
      - 1            (28 images)
      - 2            (28 images)
      - ...
```

* 참고 사항
  * 이미지의 전체 색을 검은색 & 흰색에 가깝게 변환 : ```convert_standard_ocr_dataset.py```

## 4. branch 정보

* 일정 : 2025.08.24 (일) 1 day
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❌ (FAILED)

| 계획 내용                   | branch                                                    | 상태 |
|-------------------------|-----------------------------------------------------------|----|
| 데이터셋 선택                 |                                                           | ✅  |
| 데이터셋 변형                 | ```IP-special-ocr-experiment-001-transform-dataset```     | ✅  |
| 이미지 획득 (색 변환)           |                                                           | ✅  |
| 이미지 회전 각도 파악 모델 학습      | ```IP-special-ocr-experiment-002-angle-model```           | 💨 |
| 이미지 회전 처리               | ```IP-special-ocr-experiment-003-rotate```                | ⬜  |
| 전처리된 이미지에서 각 글자 영역 도출   | ```IP-special-ocr-experiment-004-extract-letter```        | ⬜  |
| 각 글자 영역에 있는 글자 분류 모델 학습 | ```IP-special-ocr-experiment-005-letter-classify-model``` | ⬜  |
| 최종 결과 도출                | ```IP-special-ocr-experiment-006-final-result```          | ⬜  |
| 실험 결과 정리                |                                                           | ⬜  |

