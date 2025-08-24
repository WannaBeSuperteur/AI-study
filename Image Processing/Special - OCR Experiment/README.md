
## 목차

* [1. 개요](#1-개요)
* [2. 디렉토리 구성](#2-디렉토리-구성)
* [3. branch 정보](#3-branch-정보)

## 1. 개요

[OCR 실험](../OCR_Experiment.md) 에 사용된 Python 코드

## 2. 디렉토리 구성

| 디렉토리                      | 설명                       |
|---------------------------|--------------------------|
| ```classify_letter```     | 각 글자 영역에 있는 글자를 분류       |
| ```get_rectangle```       | 전처리된 이미지에서 각 글자 영역 도출    |
| ```image_preprocessing``` | 이미지 전처리, 회전 각도 파악, 회전 처리 |

## 3. branch 정보

* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❌ (FAILED)

| 계획 내용                   | branch                                                    | 상태 |
|-------------------------|-----------------------------------------------------------|----|
| 데이터셋 선택                 |                                                           | 💨 |
| 이미지 획득 (색 변환)           | ```IP-special-ocr-experiment-001-change-color```          | ⬜  |
| 이미지 회전 각도 파악 모델 학습      | ```IP-special-ocr-experiment-002-angle-model```           | ⬜  |
| 이미지 회전 처리               | ```IP-special-ocr-experiment-003-rotate```                | ⬜  |
| 전처리된 이미지에서 각 글자 영역 도출   | ```IP-special-ocr-experiment-004-extract-letter```        | ⬜  |
| 각 글자 영역에 있는 글자 분류 모델 학습 | ```IP-special-ocr-experiment-005-letter-classify-model``` | ⬜  |
| 최종 결과 도출                | ```IP-special-ocr-experiment-006-final-result```          | ⬜  |
| 실험 결과 정리                |                                                           | ⬜  |

