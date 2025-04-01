## 개요

* [Vision Anomaly Detection 필요성 탐구](../Special_Vision_Anomaly_Detection_필요성.md) 의 코드
* [실험 설계](../Special_Vision_Anomaly_Detection_필요성.md#2-1-실험-설계) 참고.

## 실험 실시 계획

* 전체 일정 : **2025.03.28 금 - 04.03 목 (3d)**
* 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE), ❌ (FAILED)

| 계획 내용                                                         | 일정                     | branch                                    | 상태 |
|---------------------------------------------------------------|------------------------|-------------------------------------------|----|
| 논문 선정                                                         | 03.28 금 - 03.29 토 (2d) |                                           | ✅  |
| 논문 스터디 (TinyViT + GLASS 총 2개)                                 | 03.29 토 - 03.31 월 (3d) |                                           | ✅  |
| [실험 설계](../Special_Vision_Anomaly_Detection_필요성.md#2-1-실험-설계) | 03.31 월 (1d)           |                                           | ✅  |
| Vision Classification 모델 (TinyViT) 구현                         | 04.01 화 (1d)           | ```IP-special-visionad-001-TinyViT```     | 💨 |
| Vision Anomaly Detection 모델 (GLASS) 구현                        | 04.01 화 (1d)           | ```IP-special-visionad-002-GLASS```       | ⬜  |
| 데이터셋 처리 코드 개발                                                 | 04.01 화 (1d)           | ```IP-special-visionad-003-DataProcess``` | ⬜  |
| XAI 모델 (pytorch-grad-cam) 구현                                  | 04.02 수 (1d)           | ```IP-special-visionad-004-XAI```         | ⬜  |
| 실험 1 - 정량적 성능 평가                                              | 04.02 수 (1d)           | ```IP-special-visionad-005-exp1```        | ⬜  |
| 실험 2 - 설명 능력 평가                                               | 04.02 수 (1d)           | ```IP-special-visionad-006-exp2```        | ⬜  |
| 실험 3 - 새로운 Abnormal Class 탐지 성능 평가                            | 04.03 목 (1d)           | ```IP-special-visionad-007-exp3```        | ⬜  |
| 실험 결과 정리                                                      | 04.03 목 (1d)           |                                           | ⬜  |