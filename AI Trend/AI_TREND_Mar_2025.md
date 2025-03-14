## 2025.03.15 (토)
**[데이터 고갈, '합성 데이터'는 해결책이 아니다](https://n.news.naver.com/mnews/article/277/0005561279?sid=105)** ```AI```

* 실제 데이터를 인공적으로 모방하여 생성한 **이른바 '합성 데이터'는 '모델 붕괴' (Model Collapse) 라는, AI의 성능 저하 현상** 을 일으킬 수 있다.
  * 모델 붕괴는 AI를 이용하여 생성한 데이터로 AI가 학습하는 것을 반복하다 보면 그 결과물의 품질이 갈수록 떨어지는 현상이다.
* 데이터 고갈 등으로 인해, 2030년 무렵에는 AI 학습에서 실제 인간이 만든 데이터보다 **합성 데이터의 비중이 더 높아질** 것으로 보인다. 
  * 합성 데이터는 2025년 2월 출시된 AI 모델 Grok-3이 단시간에 고성능을 달성한 비결이기도 하다.
* 합성 데이터의 장단점은 다음과 같다.
  * 결론적으로, 아래 장단점을 고려하여 **실제 데이터와 합성 데이터의 비중을 적절히 선택** 하는 것이 중요하다.  
  * 합성 데이터의 사용을 무조건 배제하는 것은 부적절하다.

| 장점                               | 단점                         |
|----------------------------------|----------------------------|
| - 실제 데이터 부족의 보완 가능<br>- 프라이버시 보호 | - 모델 붕괴 현상<br>- 데이터 편향성 증폭 |

## 2025.03.14 (금)
**[구글 딥마인드, 로보틱스 모델 발표](https://n.news.naver.com/mnews/article/092/0002366497?sid=105)** ```AI``` ```AI Robot```

* 2025.03.13일 구글 딥마인드는 **로봇이 현실 세계에서 행동할 수 있게 하는 AI 모델인 '제미나이 로보틱스'와 '제미나이 로보틱스-ER'을 공개** 했다.
  * 이 모델은 Gemini-2.0 을 기반으로 한 **비전-언어-행동 (VLA)** 모델이다.
* 해당 모델이 적용되는 로봇의 범위는 다음과 같다.
  * 알로하 2 쌍팔 로봇
  * 프랑카 기반 로봇 (연구실에서 사용)
* 해당 모델의 특징은 다음과 같다.
  * 로봇이 인간의 특징인 상호 작용성 및 기민성을 보이게 함
  * 복잡한 작업을 처리할 수 있는 정교한 조작 능력
  * 제미나이 로보틱스-ER 의 경우 **로봇이 작업을 보다 안전하고 효율적으로 할 수 있는 방법을 스스로 판단** 하게 할 수 있음
    * 인식은 물론 공간 이해, 코드 생성 등까지 가능
* 구글 딥마인드는 **AI가 실제 물리적 현실에서 도움을 주려면 추론 능력은 물론이고 정교한 행동이 필수** 라고 말했다.

## 2025.03.13 (목)
**[KAIST x 삼성전자, LLM 학습 시간 단축 시뮬레이션 기술 공개](https://n.news.naver.com/mnews/article/001/0015262374?sid=105)** ```AI``` ```Large Language Model```

* KAIST와 삼성전자가 협력하여 **거대 언어 모델 (LLM) 의 학습 시간을 예측하고 이를 최적화하기 위한 시뮬레이션 기술을 개발하여 공개** 했다.
  * 해당 기술은 Github에 [오픈소스](https://github.com/VIA-Research/vTrain) 로 공개되었다.
* 해결하고자 한 문제 및 그 해결 방법은 다음과 같다.

| 구분    | 설명                                                                                                                   |
|-------|----------------------------------------------------------------------------------------------------------------------|
| 문제    | 학습 효율을 높이기 위한 분산 학습 전략 탐색에 막대한 자원 (비용, 시간, 에너지 등) 이 소비됨<br>- 현업에서는 이를 해결하기 위한 현실적인 방법으로, **경험적으로 검증된 소수의 방법들** 만을 이용 |
| 해결 방법 | LLM의 학습 시간 예측 및 전략 탐색 시뮬레이션 프레임워크 'vTrain'                                                                           |

* vTrain 의 성능 수준은 다음과 같다.
  * LLM 의 실제 학습 시간과 vTrain 의 예측값의 차이가 **단일 노드 평균 절대 오차 (MAPE) 8.37%, 다중 노드에서는 14.73%** 의 높은 정확도 기록 
  * 기존 방식 대비 **GPU 사용률 10% 이상 향상 & 학습 비용 5% 이상 절감**

## 2025.03.12 (수)
**[네이버, AI 쇼핑 앱 '네이버 플러스 스토어' 출시](https://n.news.naver.com/mnews/article/015/0005105339?sid=105)** ```AI``` ```Naver AI```

* 네이버가 2025.03.12일 AI를 적용한 쇼핑 앱인 '네이버 플러스 스토어'를 출시했다.
  * 이는 OpenAI가 '오퍼레이터'라는 AI 에이전트를 최근 출시하며 쇼핑 분야에 진출하는 등 **빅테크 기업들의 위협에 대응** 하기 위한 것으로 보인다.
* 네이버 플러스 스토어에 적용된 AI 기술은 다음과 같다.
  * 네이버의 자체 LLM 하이퍼클로바X 적용
  * 이용자의 패턴 (선호도, 과거 기록, 검색 의도 등) 을 AI가 분석하여 맞춤형 정보 제공
  * '발견' 서비스의 숏폼 동영상에도 AI 적용
* 네이버 플러스 스토어에 적용된 모델을 비롯한 AI가 사용자를 위한 맞춤형 추천을 **우수한 성능으로** 할 수 있게 된다면, **AI 쇼핑이 쇼핑의 판도를 바꿀** 것으로 예상된다.
* 한편, 기존 e커머스 업체들이 쇼핑 분야에 AI를 적용하기를 망설인 이유는 **AI가 사용자의 니즈를 오판하여 잘못된 추천을 할 수 있기** 때문이었다.

## 2025.03.11 (화)
**[ChatGPT, 전 연령대에서 생성형 AI 중 추정 사용률 1위](https://n.news.naver.com/mnews/article/001/0015257960?sid=105)** ```AI``` ```Generative AI``` ```Large Language Model``` ```ChatGPT```

* ChatGPT가 2025.03.03일부터 03.09일까지를 기준으로 **모든 연령대에서 추정 사용자 수가 가장 많은 생성형 AI 앱** 인 것으로 나타났다.
  * ChatGPT의 최다 사용 연령대는 20대로, 사용자 수는 **약 1,905,700명** 으로 추정된다.
* 2위부터는 다음과 같이 연령대별로 다른 양상을 보였다.
  * 딥시크는 1월 말 조사에서는 ChatGPT에 이은 2위로 기록되었으나, 그 이후 순위가 급락한 셈이다.

| 생성형 AI 앱           | 양상                                  |
|--------------------|-------------------------------------|
| 뤼튼                 | - 30대 이하에서 높음 (최고 2위)               |
| 에이닷                | - 40대 이상에서 높음 (최고 2위)               |
| 퍼플렉시티 (Perplexity) | - 전 연령대에서 3~4위 수준                   |
| 딥시크 (DeepSeek)     | - 전 연령대에서 5~6위 수준<br>- 단, 20대에서는 7위 |

## 2025.03.10 (월)
**[정부, 2025년 APEC 핵심 성과로 AI 제시](https://n.news.naver.com/mnews/article/003/0013108291?sid=100)** ```AI```

* 정부가 2025.02.24일 열린 APEC (아시아태평양경제협력체) 회의에서 2025년에 추진할 핵심 성과 중 하나로 **AI (인공지능)** 를 제시했다.
* AI 분야에서 추진할 핵심 성과는 다음과 같다.
  * 그 명칭은 'AI 협력'
  * AI 기술의 잠재력에 대한, 지속 가능한 방향으로의 구현 방향성
* APEC 회원국들은 AI를 포함해 대한민국이 제시한 2025년 목표 핵심 성과들에 대해 시의 적절한 주제라고 평가했다.

## 2025.03.09 (일)
**[제 2의 딥시크? AI 비서 '마누스' 공개](https://n.news.naver.com/mnews/article/032/0003355484?sid=105)** ```AI``` ```AI Agent```

* 2025년 1월 DeepSeek에 이어 이번에는 **2025.03.05일 공개된 마누스 (Manus)** 라는 AI 에이전트가 등장했다.
  * 해당 모델 역시 DeepSeek와 같이 저비용으로 높은 성능을 구현했다는 점에서 '제2의 딥시크'라고 할 수 있다.
* 마누스의 성능 수준은 다음과 같다.
  * 문제 해결 능력 평가 벤치마크에서 **OpenAI의 AI 에이전트 '딥 리서치'보다 좋은 수준의 성능** 달성
  * 여행 계획 등 복잡한 업무 수행 가능
* 한편, 이처럼 우수한 성능을 보여주고 있는 마누스가 딥시크 수준으로 화제가 될 수도 있을 것으로 보인다.

## 2025.03.08 (토)
**[스타트업 디노티시아, 새 한국어 추론 모델 공개](https://n.news.naver.com/mnews/article/008/0005162402?sid=101)** ```AI``` ```Large Language Model```

* 2025.03.07일 AI 스타트업 '디노티시아'가 한국어로 된 수학 및 코딩 문제 등을 해결할 수 있는 **추론형 거대 언어 모델 (LLM) 'DNA-R1'** 을 개발했다고 밝혔다.
  * DNA-R1 의 의의는 **추론 과정 전체를 한국어로 출력하는 첫 LLM** 이라는 것이다.
  * 2024년에 디노티시아에서 출시한 'DNA:디노티시아AI' 모델에서 추론 기능을 강화한 모델이다.
* 해당 모델의 성능은 다음과 같다.
  * 한국어 [LLM 성능 평가 벤치마크](../AI%20Basics/LLM%20Basics/LLM_기초_LLM의_성능_평가.md#4-llm-벤치마크-데이터셋) 인 **'KMMLU' 기준 59.9%** 의 성능으로, 동급 모델의 50.5% 수준에서 크게 향상되었다. 
* 디노티시아의 학습 방법은 다음과 같다.
  * 문장 이해력 및 논리적 사고력 극대화를 위해 대량의 한국어 데이터 학습
  * 강화학습을 통해 AI가 스스로 정답을 도출하는 추론 능력 향상
* [외부 링크: HuggingFace 의 DNA-R1 모델](https://huggingface.co/dnotitia/DNA-R1)

## 2025.03.07 (금)
**[구글 창업자 래리 페이지, AI 스타트업 설립](https://n.news.naver.com/mnews/article/011/0004458385?sid=105)** ```AI```

* 구글의 공동 창업자 중 한 명인 '래리 페이지'가 **다이나토믹스 (Dynatomics)** 라는 AI 스타트업을 창업했다.
  * 사업 분야는 제조업이며, 목표는 제조업의 혁신이다.
* 사업 방향성에 대해서는, AI를 이용한 공산품 디자인 최적화 및 이를 통한 성능 향상으로 **추정된다.**  
* 물리적인 물건의 **실제 제작** 에 AI를 사용한다는 점에서, **피지컬 AI가 AI의 최신 트렌드** 임을 다시 한 번 실감하게 한다.
* 한편, IT 업계 관계자 중 한 명은 **AI 로봇을 이용해 공정을 자동화하면 기존 (인간 중심) 과는 생산 동선 등이 달라질 것** 이라고 말했다.

## 2025.03.06 (목)
**[OpenAI, GPT-4.5 월 20달러 이용자에게도 제공](https://n.news.naver.com/mnews/article/421/0008115264?sid=105)** ```AI``` ```OpenAI``` ```ChatGPT```

* OpenAI가 2025.03.05일 (현지시각) 자사의 최신 AI 모델인 GPT-4.5를 ChatGPT의 **월 20달러 사용자 (ChatGPT Plus 사용자) 에게도 제공하기 시작** 했다.
  * 이는 해당 모델이 리서치 프리뷰 및 ChatGPT Pro (월 200달러) 사용자에게 출시된 지 불과 약 일주일 만이다.
  * GPT-4.5는 OpenAI의 **마지막 비추론 모델** 이 될 것으로 보인다.
* GPT-4.5 는 **기존 GPT-4o 모델을 대체할 수 없는데,** 그 이유는 **비용 효율성이 극도로 나쁘기** 때문이다.
* 한편, OpenAI의 GPT-5 출시 계획은 다음과 같다.
  * 추론 + 비추론 (일반) 모델 통합 (하이브리드 모델)
  * 출시 시기는 2025년 5월 예상
  * 이후 GPT-4.5 는 비용 효율성 문제로 지원 중단 가능성이 높음
* 또한, 샘 올트먼 OpenAI CEO는 **GPT-4.5 는 뭔가 다른 종류의 지능으로, 마법 같은 것을 느꼈다** 고 했다.

## 2025.03.05 (수)
**[아마존, 추론형 AI 모델 개발 추진](https://n.news.naver.com/mnews/article/009/0005453764?sid=105)** ```AI``` ```Large Language Model```

* 아마존이 OpenAI의 'o3' 또는 딥시크의 'R1'과 같이 **복잡한 추론을 할 수 있는 추론형 모델** 의 개발에 나선다.
  * 현재 개발 중인 추론 모델은 이르면 2025년 6월에 자사 AI 모델 시리즈 '노바' 를 통해 출시될 전망이다.
* 아마존의 추론형 AI 컨셉은 다음과 같다.
  * 사고 과정이 단계적이기 때문에 시간은 다소 소요될 수 있음
  * 수학, 과학 등 복잡한 문제 해결 성능이 뛰어남
  * 일반 모델 + 추론 모델의 결합인 **하이브리드 모델** 컨셉으로 예정
    * 현재는 앤스로픽의 '클로드-3.7 소넷'이 이 형태임 
* 한편, 아마존은 다음과 같은 모델을 2025년 중 출시 예정이다.

| 모델           | 핵심 아이디어                                 |
|--------------|-----------------------------------------|
| 노바 스피치 투 스피치 | 실시간 음성 대화 가능                            |
| 애니 투 애니      | 멀티모달의 궁극 (텍스트, 이미지를 포함한 다양한 형태의 데이터 활용) |

## 2025.03.04 (화)
**[슈퍼마리오를 가장 잘 하는 AI](https://n.news.naver.com/mnews/article/003/0013099490?sid=105)** ```AI```

* 미국의 한 대학에서 4가지의 AI 모델을 대상으로 **슈퍼마리오 게임에서의 생존 시간** 을 측정했고, 그 결과 **앤트로픽에서 공개한 클로드-3.7 모델** 이 최고의 성능을 보였다.
* 성능 순위는 다음과 같다.
  * 클로드-3.7 (앤트로픽) > 클로드-3.5 (앤트로픽) > GPT-4o (OpenAI), Gemini-1.5 Pro (Google) 
  * 클로드-3.7은 45초 동안 생존했으며, 적을 해치우는 모습까지 보였다.
* 이 연구에서는 **추론 모델은 게임에 적합하지 않다** 는 추가적인 결론을 얻었다.
  * 추론 모델이 행동을 결정하는 데 걸리는 시간 동안, 마리오 게임에서는 1초 만에 게임 오버가 발생할 수 있기 때문이다.
* 한편, AI의 성능을 게임을 통해 측정하려는 시도는 슈퍼마리오뿐만 아니라 로블록스 등 다른 게임에서도 이루어지고 있다.

## 2025.03.03 (월)
**[LG유플러스 인공지능 익시 (ixi)](https://n.news.naver.com/mnews/article/003/0013096541?sid=105)** ```AI```

* LG유플러스는 자사의 AI 서비스인 익시 (ixi) 가 활약하는 2050년의 미래 생활 모습을 나타낸 **익시 퓨처빌리지 (ixi FutureVillage)** 를 공개했다.
  * 익시 퓨처빌리지의 주요 컨셉은 **안심 지능 (Assured Intelligence)** 이다.
  * 익시 퓨처빌리지는 이동통신 분야의 세계 최대 전시인 'MWC25'의 전시장 중앙에 마련되었다. 
* 익시의 AI 기술 및 서비스는 다음과 같다.

| 기술/서비스      | 설명                                                          |
|-------------|-------------------------------------------------------------|
| 익시 비전       | 집안의 안전 관리를 담당하는 AI 기반의 영상 분석 솔루션                            |
| 익시오 (ixi-O) | 개인 **AI 에이전트**<br>- 고객 분석을 통한 마케팅 전략 제안, 자동화된 고객 예약 관리 등 가능 |

## 2025.03.02 (일)
**[GPU 부족, OpenAI GPT 새 모델 개발 지장](https://n.news.naver.com/mnews/article/092/0002365115?sid=105)** ```AI``` ```OpenAI``` ```ChatGPT```

* 샘 올트먼 OpenAI CEO는 X (트위터) 를 통해 **현재 OpenAI가 보유한 GPU가 부족** 하다고 했다.
  * GPU 부족의 원인은 수요 급증 및 관련 예측의 어려움 때문이다.
* GPU 부족을 해결하기 위해 OpenAI는 **자체적인 AI 칩 개발** 도 고려 중이다. 
* 이러한 GPU 부족 사태는 AI 인프라를 구축하기 위한 경쟁을 가속화시킬 것으로 전망된다.
* 한편, OpenAI의 GPT-4.5 출시 일정은 다음과 같다.
  * 이번 주부터 ChatGPT Pro (월 200달러 플랜) 사용자에게 출시
  * 다음 주부터 ChatGPT Plus 등 (월 20달러 플랜) 사용자에게 확대 제공

## 2025.03.01 (토)
**[방송통신위원회, 생성형 AI 가이드라인 발표](https://n.news.naver.com/mnews/article/003/0013094329?sid=105)** ```AI``` ```Generative AI```

* 2025.02.28일 방송통신위원회가 생성형 AI 서비스에 의한 피해 방지를 위한 가이드라인인 **생성형 AI 서비스 이용자 보호 가이드라인** 을 발표했다.
  * 이 가이드라인은 2025.03.28일부터 시행될 예정이다.
* 이 가이드라인의 구성은 다음과 같다.
  * 생성형 AI가 그 서비스 전반에 걸쳐 추구해야 할 기본 원칙 4가지
  * 해당 원칙들을 실현하기 위한 실행 방식 6가지
* 실행 방식의 구체적인 내용은 다음과 같다.
  * 실행 방식과 함께, 실제 서비스 중 이용자 보호 관련 영역에 대해서는 그 모범 사례도 제공한다.

| 실행 방식             | 상세                                                                               |
|-------------------|----------------------------------------------------------------------------------|
| 이용자 인격권 보호        | - 생성형 AI에 의해 생성된 데이터에서 인격권 침해 요소 발견 시, 이를 통제하는 알고리즘 구축 필요                        |
| AI 기반 결정 과정 고지 노력 | - 생성형 AI의 결정 과정에 대한 정보를 이용자에게 제공<br>- 생성형 AI에 의한 결과물은 AI에 의해 생성되었음을 명확히 고지 필요    |
| 그 외               | - 다양성 존중 노력<br>- 입력 데이터 수집/활용 과정 관리<br>- 문제 해결을 위한 책임 및 참여<br>- 건전한 유통/배포를 위한 노력 |

* 한편, 방송통신위원회는 2년마다 이 가이드라인을 검토하여 개선 등 조치할 계획이다.