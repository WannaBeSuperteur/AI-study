
## 목차

* [1. 개요](#1-개요)
* [2. 디렉토리 구성](#2-디렉토리-구성)
* [3. 데이터셋 구조](#3-데이터셋-구조)
  * [3-1. Scanned Images Dataset for OCR and VLM finetuning](#3-1-scanned-images-dataset-for-ocr-and-vlm-finetuning)
  * [3-2. Standard OCR Dataset](#3-2-standard-ocr-dataset) 
* [4. branch 정보](#4-branch-정보)

## 1. 개요

[OCR 실험](../OCR_Experiment.md) 에 사용된 Python 코드

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
  - train (80%)
    - Letter
    - Memo
    - Report
  - test (20%)
    - Letter
    - Memo
    - Report
```

* 참고 사항
  * 정방향 (회전되지 않은) 이 아닌 이미지는 AI 모델 없이 수작업으로 삭제 처리
  * 남아 있는 정방향 이미지만을 학습/테스트 데이터로 사용

### 3-2. Standard OCR Dataset

* 원본 데이터셋
  * [Standard OCR Dataset (from Kaggle)](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset) 
* 데이터셋 구조

```
- Special - OCR Experiment/standard_ocr_dataset
  - data
    - training_data
      - 0
      - 1
      - 2
      - ...
    - testing_data
      - 0
      - 1
      - 2
      - ...
```

## 4. branch 정보

* 일정 : 2025.08.24 (일) 1 day
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❌ (FAILED)

| 계획 내용                   | branch                                                    | 상태 |
|-------------------------|-----------------------------------------------------------|----|
| 데이터셋 선택                 |                                                           | ✅  |
| 데이터셋 변형                 | ```IP-special-ocr-experiment-001-transform-dataset```     | ⬜  |
| 이미지 획득 (색 변환)           | ```IP-special-ocr-experiment-002-change-color```          | ⬜  |
| 이미지 회전 각도 파악 모델 학습      | ```IP-special-ocr-experiment-003-angle-model```           | ⬜  |
| 이미지 회전 처리               | ```IP-special-ocr-experiment-004-rotate```                | ⬜  |
| 전처리된 이미지에서 각 글자 영역 도출   | ```IP-special-ocr-experiment-005-extract-letter```        | ⬜  |
| 각 글자 영역에 있는 글자 분류 모델 학습 | ```IP-special-ocr-experiment-006-letter-classify-model``` | ⬜  |
| 최종 결과 도출                | ```IP-special-ocr-experiment-007-final-result```          | ⬜  |
| 실험 결과 정리                |                                                           | ⬜  |

