# AI-study

* **2023.11.14 ~ 2023.11.16**
  * 카테고리화 (기초 지식, 트렌드, 생성형AI, 이미지 프로세싱(CNN, ...), NLP(자연어 처리), 캐글 등)

----

* **2023.11.17 ~ 2024.04.21**
  * 본격 학습 진행 1차 (기초 및 프로젝트 위주)

----

* **2025.01.27 ~ 03.11 : 🛠 ML 및 Vision 분야 기초 재정립**
  * 2024년 한 해 동안 AI 연구원으로 재직하면서 **AI 기초를 확실히 아는** 것의 중요성을 깨달음
  * 그동안 알고 있었던 AI 기초를 확실하게 정리
  * 머신러닝, 딥러닝의 기본 원리에 대해 **왜? 라고 질문하는** 것을 통해 보다 deep dive 를 시도
  * **딥시크 개인정보 논란 지속으로 계획 지연 (딥시크를 깊이 연구하는 것이 적절한가?)**
    * 결론: **딥시크를 깊이 연구하는 것이 적절하다.**
    * 일부 논란은 있었지만, AI 시장에 충분히 큰 영향력이 있음
    * 딥시크 앱/서비스가 아닌 모델 자체는 개인정보 논란이 없음

| 구분                                                                                                                        | 상세                        | 일정                      |
|---------------------------------------------------------------------------------------------------------------------------|---------------------------|-------------------------|
| 전체적인 계획                                                                                                                   | 전체적인 계획 수립 (딥시크 영향 대응 고려) | 2025.01.27 - 02.20      |
| [Data Science Basics](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Basics/Data%20Science%20Basics)         | 내용 정리 및 필요 내용 추가          | 2025.02.20 - 02.22 (2d) |
| [Machine Learning Models](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Basics/Machine%20Learning%20Models) | 내용 정리 및 필요 내용 추가          | 2025.02.23 - 02.27 (5d) |
| [Deep Learning Basics](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Basics/Deep%20Learning%20Basics)       | 내용 정리 및 필요 내용 추가          | 2025.02.27 - 03.05 (7d) |
| [LLM Basics](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Basics/LLM%20Basics)                             | 필요 내용 추가                  | 2025.03.06 - 03.08 (3d) |
| [Image Processing (Vision)](https://github.com/WannaBeSuperteur/AI-study/tree/main/Image%20Processing)                    | 내용 정리 및 필요 내용 추가          | 2025.03.08 - 03.11 (4d) |

----

* **2025.03.12 ~ 2025.03.17 (예정) : 🐋 DeepSeek 모델 Deep Dive 및 미니 프로젝트**
  * 비교적 저렴한 비용으로 딥러닝을 가능하게 한 DeepSeek의 모델은 2025년 AI 시장의 판도를 바꿀 가능성이 높음
  * V3, R1 등 거대 언어 모델 (LLM)
  * 야누스 프로 등 이미지 생성 모델

| 구분                    | 상세                                                          | 일정                      |
|-----------------------|-------------------------------------------------------------|-------------------------|
| DeepSeek 모델 Deep Dive | DeepSeek의 생성형 모델에 대해 집중 탐구                                  | 2025.03.12 - 03.13 (2d) |
| 미니 프로젝트               | DeepSeek의 모델을 Fine-tuning 하여 미니 서비스 개발<br>- 언어 모델 이용 서비스 1개 | 2025.03.14 - 03.17 (4d) |

----

* **2025.03.18 ~ 2025.03.25 (예정) : 🔍 비전 이상탐지 필요성 탐구**
  * Vision 분야에서, normal/abnormal 의 Classification (분류) 을 할 수 있는데 **왜 굳이 Anomaly Detection (이상 탐지) 계열의 모델이 필요할까?**
  * 위 질문에 대한 명쾌한 답을 얻기 위해 집중 탐구
  * [ChatGPT에 질문한 결과](https://chatgpt.com/share/67974281-7fb8-8010-9a1a-4b56c060e71b) 다음과 같은 답을 얻었지만, 상세한 이유에 대한 추가 탐구 필요
    * abnormal 데이터의 희소성 관련 문제 (데이터 불균형 등)
    * 다양하고 새로운 abnormal data 의 존재 가능성
    * 기타 (라벨링 비용 및 환경 문제)

| 구분    | 상세                                                                            | 일정                      |
|-------|-------------------------------------------------------------------------------|-------------------------|
| 논문 탐독 | - Vision Classification 분야 최신 모델 1개<br>- Vision Anomaly Detection 분야 최신 모델 1개 | 2025.03.18 - 03.19 (2d) |
| 실험 진행 | - 위 ChatGPT에 질문한 결과로 얻은 답에 대해 실제 데이터셋을 이용 실험                                  | 2025.03.20 - 03.25 (6d) |

----

* **2025.03.26 ~ 2025.06.14 (예정) : ⚙ ML-ops 프로젝트**
  * **SPECIAL PROJECT I : ML-ops 프로젝트 개발**
    * **공개된 모델을 개선** 하는 모델을 여러 개 만들고, 이들을 포함하는 컨셉
    * [AI Trend](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Trend) 를 참고하여 주제 선정 
  * 2024년 한 해 동안 AI 연구원으로 재직하면서 얻은 교훈을 반영
    * 공개된 모델 구현 시, 코드가 오류 없이 돌아간다고 모델이 정상 작동하는 것을 보장할 수 없음
  * 마찬가지로, **왜? 라는 질문** 을 통해 보다 deep dive 를 시도

| 구분       | 상세                                                                                                     | 일정                       |
|----------|--------------------------------------------------------------------------------------------------------|--------------------------|
| 기본 뼈대 개발 | 여러 개의 모델을 호출 가능한 기본 뼈대 계획 및 개발                                                                         | 2025.03.26 - 03.31 (6d)  |
| 모델 1     | TBU ([AI Trend](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Trend) 를 포함한 AI 업계 트렌드 고려) | 2025.04.01 - 04.17 (17d) |
| 모델 2     | TBU ([AI Trend](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Trend) 를 포함한 AI 업계 트렌드 고려) | 2025.04.18 - 05.06 (19d) |
| 모델 3     | TBU ([AI Trend](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Trend) 를 포함한 AI 업계 트렌드 고려) | 2025.05.07 - 05.24 (18d) |
| 모델 4     | TBU ([AI Trend](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Trend) 를 포함한 AI 업계 트렌드 고려) | 2025.05.25 - 06.08 (15d) |
| 최종 정리    |                                                                                                        | 2025.06.09 - 06.14 (6d)  | 

----

* **2025.06.15 ~ 2025.10.31 (최대 2025.12.31 까지 연장 가능)**
  * 본격 학습 진행 2차
  * Generative AI, Image Processing, NLP 각각 아이디어 도출 및 미니 프로젝트 진행
  * **SPECIAL PROJECT I 진행 상황 및 결과에 기반하여 상세 일정 계획 예정**
  * 상태 : ⬜ (TODO), 💨 (ING), ✅ (DONE)
  * 중요도 : 🔴 (mandatory), 🟠 (high), 🟡 (medium), 🟢 (low)

| Field            | 프로젝트 ID | 프로젝트 요약 | 중요도 | 아이디어 도출일   | 프로젝트 기간            | 상태 |
|------------------|---------|---------|-----|------------|--------------------|----|
| Generative AI    | GAI-P3  |         |     | 2025.mm.dd | 2025.mm.dd - mm.dd | ⬜  |
| Image Processing | IP-P4   |         |     | 2025.mm.dd | 2025.mm.dd - mm.dd | ⬜  |
| NLP              | NLP-P7  |         |     | 2025.mm.dd | 2025.mm.dd - mm.dd | ⬜  |

