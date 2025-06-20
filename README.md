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

* **2025.03.12 ~ 2025.03.27 : 🐋 DeepSeek 모델 Deep Dive 및 미니 프로젝트**
  * 비교적 저렴한 비용으로 딥러닝을 가능하게 한 DeepSeek의 모델은 2025년 AI 시장의 판도를 바꿀 가능성이 높음
  * V3, R1 등 거대 언어 모델 (LLM) 관련 프로젝트
  * [프로젝트 repo](https://github.com/WannaBeSuperteur/AI_Projects/tree/main/2025_03_12_DeepSeek_LLM)

| 구분                    | 상세                                                          | 일정                       |
|-----------------------|-------------------------------------------------------------|--------------------------|
| DeepSeek 모델 Deep Dive | DeepSeek의 생성형 모델에 대해 집중 탐구                                  | 2025.03.12 - 03.13 (2d)  |
| 미니 프로젝트               | DeepSeek의 모델을 Fine-tuning 하여 미니 서비스 개발<br>- 언어 모델 이용 서비스 1개 | 2025.03.14 - 03.27 (14d) |

----

* **2025.03.28 ~ 2025.04.08 : 🔍 비전 이상탐지 필요성 탐구**
  * Vision 분야에서, normal/abnormal 의 Classification (분류) 을 할 수 있는데 **왜 굳이 Anomaly Detection (이상 탐지) 계열의 모델이 필요할까?**
  * 위 질문에 대한 명쾌한 답을 얻기 위해 집중 탐구
  * [ChatGPT에 질문한 결과](https://chatgpt.com/share/67974281-7fb8-8010-9a1a-4b56c060e71b) 다음과 같은 답을 얻었지만, 상세한 이유에 대한 추가 탐구 필요
    * abnormal 데이터의 희소성 관련 문제 (데이터 불균형 등)
    * 다양하고 새로운 abnormal data 의 존재 가능성
    * 기타 (라벨링 비용 및 환경 문제)
  * [문서 링크](Image%20Processing/Special_Vision_Anomaly_Detection_필요성.md)

| 구분    | 상세                                                                            | 일정                      |
|-------|-------------------------------------------------------------------------------|-------------------------|
| 논문 탐독 | - Vision Classification 분야 최신 모델 1개<br>- Vision Anomaly Detection 분야 최신 모델 1개 | 2025.03.28 - 03.31 (4d) |
| 실험 진행 | - 위 ChatGPT에 질문한 결과로 얻은 답에 대해 실제 데이터셋을 이용 실험                                  | 2025.03.31 - 04.08 (9d) |

----

* **2025.04.08 ~ 2025.07.17 (예정) : 본격 프로젝트**
  * [AI Trend](https://github.com/WannaBeSuperteur/AI-study/tree/main/AI%20Trend) 를 참고하여 주제 선정 
  * 2024년 한 해 동안 AI 연구원으로 재직하면서 얻은 교훈을 반영
    * 공개된 모델 구현 시, 코드가 오류 없이 돌아간다고 모델이 정상 작동하는 것을 보장할 수 없음
  * 마찬가지로, **왜? 라는 질문** 을 통해 보다 deep dive 를 시도

| 구분     | 링크                                                                                                      | 상세                                                                                                                                                                                                                                        | 일정                       |
|--------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| 프로젝트 1 | [Project Link](https://github.com/WannaBeSuperteur/AI_Projects/tree/main/2025_04_08_OhLoRA)             | **Oh-LoRA v1**<br>AI 가상 인간 **Oh-LoRA 👱‍♀️ (오로라)** 생성<br>- Oh-LoRA 의 얼굴 이미지 생성<br>- LLM 을 이용한 사용자 대화 구현                                                                                                                                   | 2025.04.08 - 04.25 (18d) |
| 프로젝트 2 | [Project Link](https://github.com/WannaBeSuperteur/AI_Projects/tree/main/2025_05_02_OhLoRA_v2)          | **Oh-LoRA v2**<br>AI 가상 인간 **Oh-LoRA 👱‍♀️ (오로라)** 생성 고도화<br>- Oh-LoRA 의 얼굴 이미지 생성 고도화<br>- LLM 을 이용한 사용자 대화 구현 고도화                                                                                                                       | 2025.05.02 - 05.21 (20d) |
| 프로젝트 3 | [Project Link](https://github.com/WannaBeSuperteur/AI_Projects/tree/main/2025_05_22_Improve_EffiSegNet) | 의료 이미지 (위장관 용종 이미지) Segmentation<br>- **기존 모델 (EffiSegNet) 개선**                                                                                                                                                                           | 2025.05.22 - 05.26 (5d)  |
| 프로젝트 4 | [Project Link](https://github.com/WannaBeSuperteur/AI_Projects/tree/main/2025_05_26_OhLoRA_v3)          | **Oh-LoRA v3**<br>AI 가상 인간 **Oh-LoRA 👱‍♀️ (오로라)** 생성 고도화<br>- Oh-LoRA 의 얼굴 이미지 생성 고도화<br>- LLM 을 이용한 사용자 대화 구현 고도화                                                                                                                       | 2025.05.26 - 06.05 (11d) |
| 프로젝트 5 | [Project Link](https://github.com/WannaBeSuperteur/AI_Projects/tree/main/2025_06_07_OhLoRA_v3_1)        | **Oh-LoRA v3.1**<br>AI 가상 인간 **Oh-LoRA 👱‍♀️ (오로라)** 얼굴 이미지 생성 고도화                                                                                                                                                                        | 2025.06.07 - 06.13 (7d)  |
| 프로젝트 6 |                                                                                                         | **Oh-LoRA v4**<br>AI 가상 인간 **Oh-LoRA 👱‍♀️ (오로라)** LLM 의 AI 윤리 메커니즘 개선                                                                                                                                                                    | 2025.06.22 - 06.28 (7d)  |
| 프로젝트 7 |                                                                                                         | **Oh-LoRA AI Teacher**<br>AI를 이용한 머신러닝 학습 기능 (예정)<br>- 머신러닝에 대한 Q&A 기능 (AI 지식, AI 트렌드 / [RAG](AI%20Basics/LLM%20Basics/LLM_기초_RAG.md) 이용)<br>- 머신러닝 퀴즈 출제 및 채점<br>- 실시간 코딩 도우미 (AI 에이전트 컨셉)<br>- Vision 딥러닝 모델 설계 능력을 AI와 대결 (최적 하이퍼파라미터) | 2025.06.29 - 07.17 (19d) |
