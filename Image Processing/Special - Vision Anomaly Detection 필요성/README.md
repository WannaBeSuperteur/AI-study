## 목차

* [1. 개요](#1-개요)
* [2. 실험 실시 계획](#2-실험-실시-계획)
* [3. 코드 실행 순서](#3-코드-실행-순서)

## 1. 개요

* [Vision Anomaly Detection 필요성 탐구](../Special_Vision_Anomaly_Detection_필요성.md) 의 코드
* [실험 설계](../Special_Vision_Anomaly_Detection_필요성.md#2-1-실험-설계) 참고.

## 2. 실험 실시 계획

* 전체 일정 : **2025.03.28 금 - 04.03 목 (7d)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❌ (FAILED)

| 계획 내용                                                         | 일정                     | branch                                    | 상태 |
|---------------------------------------------------------------|------------------------|-------------------------------------------|----|
| 논문 선정                                                         | 03.28 금 - 03.29 토 (2d) |                                           | ✅  |
| 논문 스터디 (TinyViT + GLASS 총 2개)                                 | 03.29 토 - 03.31 월 (3d) |                                           | ✅  |
| [실험 설계](../Special_Vision_Anomaly_Detection_필요성.md#2-1-실험-설계) | 03.31 월 (1d)           |                                           | ✅  |
| Vision Classification 모델 (TinyViT) 구현                         | 04.01 화 (1d)           | ```IP-special-visionad-001-TinyViT```     | ✅  |
| Vision Anomaly Detection 모델 (GLASS) 구현                        | 04.01 화 (1d)           | ```IP-special-visionad-002-GLASS```       | 💨 |
| 데이터셋 처리 코드 개발                                                 | 04.01 화 (1d)           | ```IP-special-visionad-003-DataProcess``` | ⬜  |
| XAI 모델 (pytorch-grad-cam) 구현                                  | 04.02 수 (1d)           | ```IP-special-visionad-004-XAI```         | ⬜  |
| 실험 1 - 정량적 성능 평가                                              | 04.02 수 (1d)           | ```IP-special-visionad-005-exp1```        | ⬜  |
| 실험 2 - 설명 능력 평가                                               | 04.02 수 (1d)           | ```IP-special-visionad-006-exp2```        | ⬜  |
| 실험 3 - 새로운 Abnormal Class 탐지 성능 평가                            | 04.03 목 (1d)           | ```IP-special-visionad-007-exp3```        | ⬜  |
| 실험 결과 정리                                                      | 04.03 목 (1d)           |                                           | ⬜  |

## 3. 코드 실행 순서

**1. 학습에 필요한 전체 데이터셋 처리**

```
python handle_dataset/main.py
```

**2. 학습 실시**

```
python train_model/train_model_common.py
```

**3. 실험 실시 (학습된 모델 기반)**

* 정량적 성능 평가, 설명 능력 평가, 새로운 Abnormal Class 탐지 성능 평가 순
* 아래에서 각 line 은 각각의 실험 코드를 실행함을 의미

```
python run_experiment/test_numeric_performance.py
python run_experiment/test_explanation.py
python run_experiment/test_new_abnormal_detect.py
```
