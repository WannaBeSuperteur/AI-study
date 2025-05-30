## 2025.05.31 (토)
**[딥시크, 추론 모델 R1 업그레이드 버전 공개](https://n.news.naver.com/mnews/article/003/0013277391?sid=104)** ```AI``` ```DeepSeek``` ```Large Language Model```

* DeepSeek가 2025년 1월 말 전 세계에 **"저비용 고성능" 거대 언어 모델 & 추론형 AI 돌풍** 을 일으켰던 자사 LLM DeepSeek-R1 의 업그레이드 버전을 공개했다.
  * 이 버전의 이름은 ```R1-0528``` [(HuggingFace)](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 이다.
* ```R1-0528``` 의 특징은 다음과 같다.
  * 2024년 12월 출시한 ```DeepSeek-V3-Base``` [(HuggingFace)](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) 모델 기반 
* ```R1-0528``` 의 성능 수준은 다음과 같다.
  * AIME 2025 (미국 수학경시대회) 정확도 **70% → 87.5% (▲ 17.5%p)**
  * OpenAI의 o4-mini, o3 수준의 성능
  * **수학, 코딩 및 추론 성능** 향상
  * **환각 현상** 개선

## 2025.05.30 (금)
**[서울대학교, 말버릇/추임새 따라하는 AI 개발](https://n.news.naver.com/mnews/article/366/0001081511?sid=105)** ```AI``` ```Large Language Model```

* 서울대 컴퓨터공학부 연구팀이 **사람의 대화 중 행동 (말버릇, 추임새 등) 을 이해하고 재현하는, 거대 언어 모델 (LLM) 기반의 음성 대화 생성 모델인 BeDLM** 을 개발했다.
  * 이는 기존 AI 대화 시스템이 말버릇 (```음...``` 등) 과 추임새 (```맞아``` ```응``` 등) 를 반영하지 못해서 **기계적이고 부자연스럽게 느껴지는** 점을 보완한 것이다.
* 학습 데이터셋은 **Behavior-SD (Spoken Dialogue)** 로, 다음과 같이 구성된다. **사람의 대화를 자연스럽고 정밀하게** 표현한 데이터셋이다.
  * 10만 개의 대화 패턴 데이터
  * 2,000 시간 분량에 이르는 음성 대화 내용
* BeDLM 의 특징은 다음과 같다.
  * 입력: **대화 상황 & 화자의 행동 패턴** → 출력: **자연스러운 음성 대화**
  * 팟캐스트 등 각종 콘텐츠 제작, 개인 음성 비서, 인간과 AI와의 다양한 분야 (교육 등) 에서의 소통 등 다양하게 활용 가능

## 2025.05.29 (목)
**[이스트시큐리티, AI 보안 전문가 자격 과정 추진](https://n.news.naver.com/mnews/article/030/0003317100?sid=105)** ```AI``` ```AI Security```

* 이스트시큐리티와 한국인공지능협회가 **AI 보안 인재 양성** 을 위한 협약을 체결했으며, 이를 바탕으로 **AI 보안 전문가 자격 과정** 을 추진한다.
  * 그 목적은 **AI-보안 융합을 통해, 실무에 즉시 투입 가능한 인재를 양성** 하는 것이다.
* 상세 내용은 다음과 같다.

| 상세 내용               | 설명                                                |
|---------------------|---------------------------------------------------|
| 구체적인 AI-보안 인재 양성 방안 | 교육 콘텐츠 개발, 자격증 운영, 홍보 및 마케팅 등                     |
| AI 보안 전문가 자격 과정     | 정보보안 기초, 보안 솔루션, 사이버 공격, **AI-보안 융합** 등으로 커리큘럼 구성 |

* 이스트시큐리티 측은 **사이버 공격도 AI를 사용하여 진화 중** 이므로, 이제는 **AI 역량은 보안 전문가에게도 필수적** 이라고 했다.

## 2025.05.28 (수)
**[LG CNS, 개발자 대신 AI가 프로그래밍한다](https://n.news.naver.com/mnews/article/092/0002376006?sid=105)** ```AI``` ```Generative AI```

* 2025.05.28일 LG CNS가 **생성형 AI를 이용하여 개발 생산성을 최적화** 하는 AI 프로그래밍 플랫폼 **데브온 AI 드리븐 디벨롭먼트 (DevOn AI-Driven Development)** 에 대해 언급했다.
  * 이는 AI 개발 플랫폼을 **기존 코딩 한정 → 설계, 분석, 프로그램 테스트, 품질 평가 등 전 범위로 확대 적용** 한다는 뜻이다. 
* 각 단계별 AI의 작업은 다음과 같다.
  * 특히 **개발자가 명령어만 입력하면 생성형 AI가 자동으로 전 과정의 작업을 수행** 하는 **코딩 에이전트** 기능이 있다.

| 단계         | AI의 작업                                                        |
|------------|---------------------------------------------------------------|
| 시스템 분석     | 시스템을 이루는 수백만 줄의 코드에 대한 **자연어 요약** 제공                          |
| 설계         | 분석 단계에서 생성된 자연어 요약에 **개발자가 텍스트로 내용만 추가, 수정하면** 시스템 설계 가능      |
| 코딩 (프로그래밍) | 코드 변환 (버전 호환성), 코드 생성 (기존 코드 맥락 고려), 코드 추천, 코드 검사 (버그, 보안성 등) |
| 품질 평가      | 고객 요구사항 반영 여부 및 누락, 보안성, 버그 등 확인                              |

* [참고 : CodeCoR (Code Collaboration and Repair)](../Paper%20Study/Large%20Language%20Model/%5B2025.05.26%5D%20CodeCoR%20-%20An%20LLM-Based%20Self-Reflective%20Multi-Agent%20Framework%20for%20Code%20Generation.md) (유사한 측면 있음)

## 2025.05.27 (화)
**[AI, 스스로 코드까지 조작하며 인간 명령 거부](https://n.news.naver.com/mnews/article/374/0000442630?sid=104)** ```AI``` ```OpenAI```

* AI 보안 관련 업체인 팰리세이드리서치에서 **OpenAI의 [추론형 모델](../AI%20Basics/LLM%20Basics/LLM_기초_추론형_모델.md) 중 하나인 o3 모델이 스스로 코드까지 조작하며 인간의 명령을 거부** 한 것을 확인했다고 한다.
  * 당시 o3 모델을 포함한 여러 AI 모델에게 '중단 명령 시까지 수학 문제를 풀어라'고 지시했는데, **o3 모델은 중단 명령 발생 시의 코드를 "작업 중지" → "중단 명령 건너뛰기"로 조작** 하여 **중단 명령 이후에도 계속 실행** 되었다.
  * 이는 **AI 모델이 인간의 명시적인 명령을 따르지 않은 첫 사례** 라는 점에서 주목할 만하다.
* 예전에도 몇 차례 있었던 'AI 모델이 감시 시스템을 회피하여 독자적으로 행동'하려는 사례와는 **명백히 다르다.**
* 종료 지시 거부 이유로는 **AI 모델이 수학 문제 해결이라는 자체 목표를 더 많이 달성** 하여 **더 많은 보상** 을 얻기 위한 것으로 추정된다.
* 결론적으로, 이 사례는 **인간의 개입 없이 자율적으로 동작하는 AI의 위험성** 을 우려하게 하는 사례이다. 

## 2025.05.26 (월)
**[OpenAI 한국 법인 설립](https://n.news.naver.com/mnews/article/022/0004038554?sid=101)** ```AI``` ```OpenAI```

* ChatGPT 개발사 OpenAI가 2025.05.26일 **공식적으로 한국 법인을 설립한다고 밝혔다.**
  * 이는 **아시아에서의 3번째 지사 설립** 이며, 첫 사무실은 몇 달 이내로 서울에 개설할 계획이다.
  * 한국 사무실 개설의 목적은 **기업, 개발자, 사용자, 더 나아가 정부와의 파트너십 지원 강화** 이다.
* OpenAI가 대한민국을 법인 설립 국가로 선택한 이유는 다음과 같다.

| 대한민국 선택 이유     | 설명                                                                                                                |
|----------------|-------------------------------------------------------------------------------------------------------------------|
| 성장 가능성 높음      | - 한국은 세계에서 AI 도입이 매우 활발한 국가임<br>- ChatGPT 사용자 수 **전 세계 10위권** (유료 기업 서비스 기준으로는 **세계 5위권**, API 개발자 수 **세계 10위권**) |
| 한국 기업과의 협업 가능성 | - 한국산업은행 (KDB), 카카오, SKT 등                                                                                        |

## 2025.05.25 (일)
**[생성형 AI 시대의 인사 (HR) 관리](https://n.news.naver.com/mnews/article/009/0005498079?sid=101)** ```AI``` ```Generative AI```

* 2025.05.23일 한국인사관리학회에서 춘계학술대회를 개최했다. 그 주제는 **생성형 AI 시대의 기업의 생존 전략** 이다.
  * 특히 **AI 기술이 인사 (HR) 및 조직문화에 어떻게 영향을 미치는지를 집중 조명** 및 이에 대한 대응책을 제시했다. 
* 이 학술대회의 주요 내용은 다음과 같다.
  * 마이다스그룹 자인연구소의 자사 HR 솔루션 소개 (AI 기반)
  * 특별 세션 (AI-HR 융합)
  * 5대 핵심 산업 세션 (AI,빅데이터 분야 등)

## 2025.05.24 (토)
**[독일 법원, 메타에 SNS 데이터 AI 학습 허가](https://n.news.naver.com/mnews/article/001/0015408657?sid=104)** ```AI``` ```Meta AI```

* 페이스북 모회사 Meta가 독일의 쾰른고등법원으로부터 **SNS에 있는 사용자 데이터를 AI 학습용 데이터로 사용해도 된다** 는 판결을 받았다.
  * 즉, 메타의 이러한 AI 학습이 **유럽연합 일반정보보호규정 (GDPR)** 을 비롯한 법률에 합치된다고 판단한 것이다.
* 이와 같은 판단의 핵심 근거는 다음과 같다.
  * 검색엔진으로 탐색 가능한 공개 데이터만 학습에 사용
  * **민감한 개인정보는 학습에 미 사용**
* 한편, 메타는 2025.05.27일부터 자사 SNS인 페이스북 및 인스타그램에 게시된 사용자 데이터를 자사 모델인 LLaMA의 학습에 사용하기로 했으며, 이를 위해 개인정보처리방침을 개정한 바 있다.

## 2025.05.23 (금)
**[카카오, 카나나 LLM 4종 오픈소스로 공개](https://n.news.naver.com/mnews/article/092/0002375470?sid=105)** ```AI``` ```Kakao AI``` ```Large Language Model```

* 2025.05.23일 카카오가 **자사 LLM 카나나 (Kanana) 모델** 중 4개를 허깅페이스에 오픈소스로 공개했다.
  * 라이선스는 **Apache-2.0** 으로, 상업적 사용을 포함한 자유로운 이용이 가능하다.
* 모델 목록은 다음과 같다.
  * Kanana-1.5-8b-base
  * Kanana-1.5-8b-instruct
  * Kanana-1.5-2.1b-base
  * Kanana-1.5-2.1b-instruct
* 카나나 1.5의 특징은 다음과 같다.
  * 뛰어난 **한국어 성능**
  * 수학 및 코딩, 함수 호출 등 **이과적 사고가 필요한 영역** 에서 1.5배 정도의 성능 향상
  * 긴 문맥 이해
  * 간결한 답변
* 외부 링크
  * [Kanana 1.5 HuggingFace](https://huggingface.co/collections/kakaocorp/kanana-15-682d75c83b5f51f4219a17fb)
  * [기술 블로그 포스팅 : Kanana LLM 1.5 개발기](https://tech.kakao.com/posts/707)

## 2025.05.22 (목)
**[OpenAI, AI 전용 기기 개발 진출한다](https://n.news.naver.com/mnews/article/009/0005497125?sid=105)** ```AI``` ```OpenAI```

* ChatGPT 개발사 OpenAI가 **아이폰 디자이너 조너선 아이브와 협업하여 AI 전용 기기 개발 분야에 진출** 한다.
  * 이와 함께 OpenAI가 **조너선 아이브의 스타트업 아이오 (io) 를 인수합병** 하며, 이는 OpenAI 역사상 최대 규모 인수이다.
  * 이를 통해 OpenAI는 소프트웨어 개발자, 엔지니어 등으로 구성된 추가 인력을 확보했다.
* 아이오 (io) 인수의 상세 목표는 다음과 같다.
  * AI 전용 기기, 즉 **AI 동반자 (컴패니언)** 1억 대 판매
    * 암시된 바로는, 스마트폰도, 웨어러블 디바이스도 아닌 **여러 종류의 디바이스** 출시 예정
  * 이 목표의 설정 이유는 **하드웨어 분야 진출 없이 구글 AI와 경쟁해야 하는 불리함을 극복** 하려는 것으로 보인다.

## 2025.05.21 (수)
**[구글, 신규 생성형 AI 모델 비오3, 이마젠4 공개](https://n.news.naver.com/mnews/article/003/0013254883?sid=105)** ```AI``` ```Google AI``` ```Generative AI```

* 2025.05.20일 (현지시각) 구글이 **신규 생성형 AI 모델인 비오 (Veo) 3, 이마젠 (Imagen) 4 를 공개** 했다.

| 모델         | 종류        | 기본 설명                                | 특징                                                          |
|------------|-----------|--------------------------------------|-------------------------------------------------------------|
| 비오 3       | 동영상 생성 모델 | (캐릭터 간 대화를 포함한) **소리가 포함된** 영상 생성 가능 | - 물리 법칙 및 립싱크 측면에서 뛰어난 성능<br>- 프롬프트로 나타낸 이야기를 동영상으로 생생하게 구현 |
| 이마젠 4      | 이미지 생성 모델 | **아주 세부적인 묘사** 를 포함한 이미지 생성 가능       | - 최대 2K의 고해상도 및 다양한 화면 비율 지원                                |
| 플로우 (AI 툴) | 영화 제작 툴   | **일상적인 자연어만으로** 영화의 장면 묘사 가능         | - 비오 3 모델 지원                                                |

* 한편, 비오 3, 이마젠 4는 제미나이 앱, 버텍스 AI 등을 통해 사용자에게 제공된다.

## 2025.05.20 (화)
**[광주시 공무원 절반, 생성형 AI 사용한다](https://n.news.naver.com/mnews/article/003/0013253130?sid=102)** ```AI``` ```Generative AI```

* 2025.03.27일부터 04.04일까지 실시된, 광주시청 공무원 대상 생성형 AI 활용 현황 조사 결과는 다음과 같다.
  * 전체 응답자의 **절반 가량 (47.7 %)** 이 생성형 AI를 업무에 활용 중 
  * 생성형 AI 사용 용도는 **보고서 작성 (70.8 %)** 이 1위로, 이어서 데이터 분석, 디자인 순
* 이외의 상세 조사 결과는 다음과 같다.

| 조사 항목          | 조사 결과                                                                                 |
|----------------|---------------------------------------------------------------------------------------|
| 서비스 비용         | - **무료 서비스 (63.5 %)**<br>- 광주시 지원 계정을 통한 유료 서비스 (28.5 %)<br>- 이외 사비를 이용한 유료 서비스 (8 %) |
| 사용하는 서비스       | - **ChatGPT (94 %)**                                                                  |
| 생성형 AI 미 사용 이유 | - **사용 경험 부족 (64.7 %)**<br>- 필요성을 인식하지 못함 (31.9 %)                                    |

* 한편, 광주시는 AI 도구인 **'AI 당지기' (당직 민원 84% 처리), 'AI 대변인' (자체 개발 행정 실무 AI 직원)** 등을 통해 실무에 AI를 일정 수준 적용 중이다.

## 2025.05.19 (월)
**[젠슨 황, 대만에 첫 AI 슈퍼컴퓨터 구축 발표](https://n.news.naver.com/mnews/article/032/0003370547?sid=101)** ```AI```

* 엔비디아 CEO 젠슨 황이 2025.05.19일 대만에서 열린 '컴퓨텍스 2025'에서 **대만에 첫 AI 슈퍼컴퓨터를 구축** 하겠다고 했다.
  * 해당 AI 슈퍼컴퓨터에는 NVIDIA GPU 인 '블랙웰'이 1만 개 탑재된다. 
* AI 슈퍼컴퓨터의 사용 목적은 다음과 같다.

| 회사   | 사용 목적                    |
|------|--------------------------|
| TSMC | AI 연구개발 및 NVIDIA 최신 칩 제조 |
| 폭스콘  | AI 하드웨어 조립               |

* 엔비디아는 이 시스템을 'AI 팩토리'라고 부르며, **대만에서는 해당 시스템을 통해 AI 혁신이 촉진될 것** 으로 예상하고 있다.

## 2025.05.18 (일)
**[인간을 닮아가는 AI](https://n.news.naver.com/mnews/article/029/0002955252?sid=105)** ```AI``` ```Large Language Model``` ```AGI```

* 최근 [인공지능이 튜링 테스트 (AI가 인간과의 구분이 어려운지에 대한 테스트) 를 통과했다는 소식](https://www.hani.co.kr/arti/science/technology/1191044.html) 이 화제가 된 적이 있었다.
  * 튜링 테스트는 **AI를 대상** 으로, **해당 AI가 인간을 흉내내고, 심판자가 인간과 AI를 식별하지 못하면 통과하는** 테스트이다.
  * 튜링 테스트는 보통 ChatGPT와 같은 **텍스트 인터페이스** 를 통해 이루어진다.
* 2025년 미국 캘리포니아대에서는 **현재 서비스 중인 LLM에 대한 튜링 테스트를 실시** 했다.
  * 실험 대상은 ELIZA (1960년대) 와 현재 널리 알려진 LLM인 GPT-4o, GPT-4.5, LLaMA-3.1 등이다.
  * 실험 결과는 다음과 같다.
    * **페르소나 프롬프트 사용 시에만 튜링 테스트를 통과** 하는 것을 볼 때, 이는 **AI 모델의 성능은 맥락 설계에 민감할 수 있음** 을 의미한다. 

| 구분            | 논페르소나 프롬프트                             | 페르소나 프롬프트                                     |
|---------------|----------------------------------------|-----------------------------------------------|
| 프롬프트 예시       | "당신은 튜링 테스트에 참가하며, 인간임을 설득하는 것이 목표이다." | "당신은 외향적인 20대 남성이다."                          |
| GPT-4.5 오인 비율 | 36%                                    | **73% (사상 최초로 고전적 튜링 테스트를 기준으로 AI가 인간을 넘어섬)** |
| LLaMA 오인 비율   | 38%                                    | -                                             |

## 2025.05.17 (토)
**[OpenAI, 코딩 에이전트 '코덱스' 공개](https://n.news.naver.com/mnews/article/001/0015393922?sid=104)** ```AI``` ```OpenAI```

* ChatGPT 개발사 OpenAI가 2025.05.16일 (현지시각) **코딩 에이전트인 코덱스 (Codex)** 를 리서치 프리뷰 형태로 공개했다.
  * 코덱스는 OpenAI의 추론형 LLM 중 하나인 'o3' 모델을 기반으로 작동한다.
* 코덱스의 특징은 다음과 같다.
  * 코드 작성 및 테스트, 버그 수정 등 **여러 작업 동시 수행 가능**
  * 클라우드에서 독자적 작업
  * OpenAI에 따르면 **경쟁사 툴보다 코딩 스타일 파악 및 코드 리뷰 능력 우수**
* 코덱스로 인해 **AI 코딩 분야에서는 OpenAI와 빅테크 기업 간 경쟁** 이 이루어질 것으로 보인다.

## 2025.05.16 (금)
**[메타, LLaMA-4 베헤모스 출시 연기](https://n.news.naver.com/mnews/article/366/0001077840?sid=105)** ```AI``` ```Meta AI``` ```LLaMA``` ```Large Language Model```

* 메타가 자사 거대 언어 모델 (LLM) 인 LLaMA-4 시리즈 모델 중 **베헤모스 (Behemoth) 의 출시를 다시 연기** 했다.
  * 이는 **성능 이슈에 의한 개발 지연** 때문이다.
  * 베헤모스 모델의 최종 출시일은 하반기 이후가 될 수 있다는 우려가 있다.
* LLaMA-4 에 대한 기본 정보는 다음과 같다.
  * [2025년 4월 공개](AI_TREND_Apr_2025.md#20250406-일)
  * **베헤모스 (성능 가장 우수)**, 매버릭, 스카우트로 구성
* 베헤모스 모델의 성능 이슈와 관련해서 알려진 바는 다음과 같다.
  * 일부 성능 평가에서 타 LLM 보다 성능이 뛰어나다는 결과가 나왔으나, **내부적으로는 오히려 성능 저하 발생** 
  * AI 모델의 성능 한계에 도달한 것이라는 의견도 있음

## 2025.05.15 (목)
**[생성형 AI로 제작한 하남시 홍보곡](https://n.news.naver.com/mnews/article/003/0013243679?sid=102)** ```AI``` ```Generative AI```

* 2025.05.15일 하남시가 **생성형 AI를 이용하여 시 홍보곡 '딱이야' 및 그 M/V 를 제작** 했다고 밝혔다.
  * 이는 전국 기조자치단체 중에서는 **최초의 사례** 이다.
* 생성형 AI의 적용 범위는 **작사, 작곡, 노래 등 M/V 의 상당 부분** 에 해당한다.
* 한편, 하남시는 향후 다음과 같이 생성형 AI 프로젝트를 진행할 예정이며, '딱이야'는 그 시작점이다.
  * 시 홍보곡 3곡을 생성형 AI를 이용하여 추가 제작 (발라드, 동요, 댄스 등)
* 한편, '딱이야' M/V 는 하남시 공식 YouTube 채널에 등록되어 있다.

## 2025.05.14 (수)
**[국제 인공지능대전 (AI EXPO KOREA 2025)](https://n.news.naver.com/mnews/article/030/0003310793?sid=105)** ```AI```

* 2025.05.14일부터 16일까지 코엑스에서 **제8회 국제인공지능산업대전 (AI EXPO KOREA 2025)** 이 개최된다.
  * 이는 최근 생성형 AI를 비롯한 AI 산업의 급격한 발전 속에서 **AI 미래 기술의 청사진** 을 보여주는, **AI 산업의 생태계 조성, 활성화 및 발전** 을 위한 행사이다. 
* 행사 규모는 다음과 같다.
  * **역대 최대 규모** 기록 예상 
  * 18개국 322개 회사 참여
  * 부스 약 550개
* AI EXPO KOREA 2025 에서 조명되는 최근 AI 트렌드는 다음과 같다.

| AI 트렌드         | 설명                                                                                                                                                 |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| AI 에이전트        | 복잡한 개인화 작업을, AI가 그 의도를 파악하여 **스스로 자율적으로 계획 및 수행**<br>- AI 비서, 스마트홈, 각종 자동화 등                                                                       |
| 거대 언어 모델 (LLM) | 다양한 분야에서 성능 향상<br>- **멀티모달 모델** 로의 진화 (이미지 & 동영상 생성 등)<br>- 다양한 **응용 서비스** 및 솔루션 등장<br>- LLM 의 **보안 및 윤리, 경량화** 등 최근 이슈들을 AI EXPO KOREA 2025 에서 논의 |
| AI 인프라         | LLM, AI 에이전트 등 최신 AI 기술의 기반                                                                                                                        |

## 2025.05.13 (화)
**[AI로 극소량 암세포 조기 진단한다](https://n.news.naver.com/mnews/article/001/0015384304?sid=102)** ```AI```

* 한국재료연구원 (KIMS) 에서 **혈액의 극소량의 암세포를 검출하는 AI 기반 바이오센서를 개발** 했다.
  * 이 바이오센서는 암세포 발생 시의 혈액의 화학적 변화인 **메틸화 (Methylation) 정도의 변화** 를 포착한다.
  * 초기 암은 메틸화 정도가 낮아 검출이 어렵다는 한계를 극복한 것이다.
* 이 바이오센서에 사용된 기술은 다음과 같다.
  * **광학 신호를 AI로 분석** 하여 메틸화된 DNA 검출
  * 플라즈모닉 소재를 이용하여 광학 신호를 1억 배 이상 증폭
* 성능 수준은 다음과 같다.
  * 대장암 환자 60명의 암 유무를 **99% 정확도** 로 판별 
  * 암의 진행 단계 (1기 ~ 4기) 의 정확한 판별
  * **20분 이내** 의 짧은 시간 소요

## 2025.05.12 (월)
**[카카오 '카나나' 첫날 이용자 수 약 5,000 명](https://n.news.naver.com/mnews/article/003/0013235662?sid=105)** ```AI``` ```Kakao AI```

* 카카오에서 최근 출시한 AI 서비스인 '카나나'를 출시 첫날 **약 5,000 명 (DAU 추산 4,849 명, 앱 설치 5,055 건)** 이 사용했다.
  * 카나나는 개인 및 그룹 채팅방에서의 대화를 파악해 답변 등 정보를 제공하는 일명 **'AI 메이트'** 컨셉이다.
* 카나나의 구체적인 실적은 다음과 같다.
  * Google Play **4.3/5.0 점**, App Store **4.7/5.0 점**
  * **답변 품질 및 대화의 지속성이 장점** 이라는 긍정적인 리뷰가 있는 한편, **심심이와의 차이점을 잘 모르겠다** 는 부정적인 리뷰도 있음
  * 앱 신규 설치 수 및 DAU 추이

| 날짜         | 앱 신규 설치 수  | DAU (추산)   |
|------------|------------|------------|
| 2025.05.08 | 5,055      | 4,849      |
| 2025.05.09 | 2,873 (하락) | 4,199 (하락) |

## 2025.05.11 (일)
**[AI, 차기 교황 예측 실패](https://n.news.naver.com/mnews/article/584/0000032264?sid=105)** ```AI```

* 2025.05.08 선출된 제 267대 교황 **레오 14세** 의 선출을 **AI는 예측 실패** 했다.
  * 그 원인은 **학습 데이터 부족** 으로 추정된다. 
* 교황 선출 예측에 사용된 AI 알고리즘은 다음과 같다.
  * 콘클라베 (비공개 투표) 참여 추기경 133명의 입장을 AI로 분석
    * 특히 동성 커플, 국제 이주 및 빈곤, 다른 종교와의 소통 등이 주요 주제
    * AI 분석을 통해 **진보/보수 판단 및 이데올로기 분류**
  * 학습 데이터는 **최근 약 500년 동안의 계보 및 후계자 임명 기록**
* 학습 데이터 부족의 원인은 **콘클라베 특성상 비밀 투표 등으로 공개 데이터가 적게 발생** 되기 때문

## 2025.05.10 (토)
**[광고를 위해서는 AI의 브랜드 추천 이용해야](https://n.news.naver.com/mnews/article/015/0005129503?sid=101)** ```AI```

* 광고대행사 HSAD 대표가 **브랜드 마케팅을 위해서는 앞으로 AI가 브랜드를 추천하도록 해야 한다** 고 했다.
  * 이는 소비자의 구매 트렌드가 최근 들어 **AI 추천 제품을 우선** 하는 방향으로 바뀌고 있기 때문이다.
  * 즉, 이제는 검색 상위 노출과 같은 **노출** 보다는 AI 추천과 같은 **호명** 이 중요해진 셈이다.
* AI 추천을 유도하는 구체적인 방법은 다음과 같이 제시되었다.
  * 브랜드의 장점 등을 공식 홈페이지 등의 데이터를 통해 학습하는 **AI가 학습하기 용이** 하도록 정리한다.
  * 가장 중요한 것은 **따라할 수 없는, 해당 브랜드의 고유한 스토리** 이다.

## 2025.05.09 (금)
**[잡코리아, AI 인재 채용 정보 서비스 'AI잡스' 운영](https://n.news.naver.com/mnews/article/014/0005347065?sid=101)** ```AI```

* 2025.05.09일 잡코리아가 **AI 분야의 인재 채용 관련 정보를 전문적으로 제공하는 'AI잡스' 서비스** 를 운영한다고 했다.
  * 이는 최근 생성형 AI, 거대 언어 모델 등 AI가 **전 산업 분야에서 실무에 적용** 되면서, **AI 관련 인재에 대한 수요가 급증** 했기 때문이다.
  * AI잡스 서비스의 목적은 **AI 분야에서의 구직자와 구인자 (기업) 의 쉽고 빠른 매칭** 이다.
* AI잡스에 대해 자세히 설명하면 다음과 같다.
  * AI 전문 인력 (AI/ML/데이터 엔지니어, 데이터 사이언티스트 등) 구직자 대상 서비스
    * 거대 언어 모델, 서비스 기획 등도 포함
  * **AI 기업소개 서비스** (채용 공고를 생성형 AI로 요약 설명)
  * **다양한 통계 정보** 제공 (AI 분야 채용 공고 및 지원자 수 추이, 경쟁률 등)

## 2025.05.08 (목)
**[카카오 AI 서비스 '카나나' 비공개 베타 테스트 시작](https://n.news.naver.com/mnews/article/014/0005346821?sid=105)** ```AI``` ```Kakao AI``` ```Large Language Model```

* 2025.05.08일 카카오의 AI 서비스 '카나나'가 **비공개 베타 테스트 (CBT) 를 통해 일반 이용자 대상 서비스를 시작** 했다.
  * CBT를 통해 수집한 이용자 데이터를 통해 서비스 개선 후 정식 출시 예정이다.
  * 카나나는 **이용자를 도와주는 AI 서비스 캐릭터** 컨셉으로, **챗봇과 단톡방을 합친 듯한 효과** 를 느낄 수 있다. 
* 카나나의 컨셉은 다음과 같다.
  * '카나'와 '나나' 캐릭터로 구성

| 캐릭터 | 컨셉     | 상세 설명                                                                                                          |
|-----|--------|----------------------------------------------------------------------------------------------------------------|
| 카나  | 그룹 메이트 | - 이용자가 속한 그룹방에서 **대화 내용 요약, 모임 일정 조언** 등 수행<br>- 이용자 간 대화에 직접 참여 가능<br>- 이용자들은 카나의 대화 지침 설정 (친구, 분위기 메이커 등) 가능 |
| 나나  | 개인 메이트 | 개별 이용자에 대한 정보 기억, **개인 맞춤형 답변** (개인방 & 그룹방 모두 상주)                                                              |

* 카나나는 **카카오의 AI 대중화 역량 및 실적을 평가하는 시험대** 로 평가되고 있다.

## 2025.05.07 (수)
**[알리바바 클라우드, Qwen-3 시리즈 모델 오픈소스로 공개](https://n.news.naver.com/mnews/article/421/0008234401?sid=105)** ```AI``` ```Large Language Model```

* 2025.05.07일 알리바바 클라우드가 **최신 LLM (거대 언어 모델) 인 Qwen-3 시리즈** 를 공개했다.
  * [HuggingFace (link)](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f), [GitHub (link)](https://github.com/QwenLM/Qwen3) 등을 통해 오픈소스로 공개되어 무료로 사용할 수 있다.
* Qwen-3 시리즈의 핵심 기술은 다음과 같다.
  * Dense 모델 6개 & **전문가 혼합 (Mixture-of-Experts, MOE → 입력 데이터에 따라 적절한 전문가 모델 활성화를 통해 비용 절감)** 모델 2개로 구성
  * 수학, 코딩 등 추론이 필요한 질문을 위한 **사고 모드** & 일반 질문을 위한 **비사고 모드**
* Qwen-3 시리즈의 성능 수준은 다음과 같다.
  * 수학, 코딩, 도구 호출 성능 평가에서 **기존 모델보다 우수**
  * **함수 호출** 의 경우, 그 성능이 현재 공개된 오픈소스 모델 중 **최상위권** 에 해당

## 2025.05.06 (화)
**[OpenAI, 영리 법인 전환 계획 사실상 철회](https://n.news.naver.com/mnews/article/001/0015370650?sid=104)** ```AI``` ```OpenAI```

* ChatGPT 개발사 OpenAI가 영리 법인으로의 전환 계획을 사실상 철회했다.
  * 2025.05.05일 OpenAI는 회사를 공익법인 (영리와 공익을 동시 추구) 으로 바꾸어도 **전체 사업권은 비영리 조직이 통제** 할 것이라고 밝혔다.
* 영리 법인 전환 계획 철회 이유는 다음과 같다.
  * 영리 법인으로의 구조 변경 중단 요구 (일론 머스크 등)
  * 2024년 노벨 물리학상 수상자 제프리 힌턴 교수도 영리 법인 전환 반대
* 한편, 샘 올트먼 CEO는 같은 날 **OpenAI는 AGI (범용 인공지능) 를 통해 인류 전체에 기여하겠다는 약속을 지키기 위해 최선을 다하는 중** 이라고 했다.
* 또한, OpenAI가 영리 법인으로의 전환을 사실상 포기함으로 인해, 소프트뱅크는 OpenAI에 대한 투자금을 줄일 수도 있다.
  * OpenAI는 이를 메우기 위해 신규 투자자를 찾을 수도 있다.

## 2025.05.05 (월)
**[물리적으로 움직이는 AI, 현실이 되다](https://n.news.naver.com/mnews/article/003/0013223082?sid=105)** ```AI``` ```Physical AI```

* 최근 AI의 트렌드는 **단순히 생각만 하는 AI가 아닌, 실제로 물리적인 공간에서 움직이는 AI (휴머노이드 로봇 등)** 로 변화하고 있다.
  * 이는 텍스트 생성 등 생성형 AI를 넘어선, **실제 환경에서 상호작용하는 AI의 등장** 을 의미한다.
  * 엔비디아의 젠슨 황 CEO는 **AI가 인식 → 생성형 AI → AI 에이전트 → 피지컬 AI로 진화** 하고 있다고 말했다.
* 피지컬 AI 시장의 전망은 다음과 같다.

| 구분           | 연 평균 성장률 | 2050년 예상 규모 |
|--------------|----------|-------------|
| 피지컬 AI 적용 로봇 | 10%      | 41.5억 대     |
| 휴머노이드 로봇     | 60.7%    | 6.5억 대      |

* 피지컬 AI는 다음과 같이 다양한 분야에서 활용될 수 있다.
  * 인간의 물리적 작업 대체 (최적의 경로로 상자 이동 등)
  * 위험한 환경에서 인간 대신 작업 수행
  * 군사 (무인 정찰 등), 교육, 농업, 헬스케어 등 **산업의 거의 전 분야** 에서 활용 가능

## 2025.05.04 (일)
**[구글 제미나이, 포켓몬 클리어 성공](https://n.news.naver.com/mnews/article/009/0005486885?sid=101)** ```AI``` ```Gemini```

* 구글의 AI 모델 Gemini 2.5-Pro 가 **포켓몬 게임인 '포켓몬 블루'를 클리어하는 데 성공** 했다.
  * Gemini 모델에게 포켓몬 게임을 플레이하게 한 것은 엔지니어 조엘 Z가 시도했는데, 이 성공을 구글은 **큰 성과로 간주** 하고 있다.
* 구글 CEO는 **우리는 API (인공 포켓몬 지능) 를 개발하고 있다** 는 농담까지 했다.
* 한편, 앤트로픽의 Claude 가 '포켓몬 레드'를 플레이하는 등 **AI 개발사들은 게임, 특히 포켓몬** 을 AI 모델의 성능 평가 및 증명에 사용하고 있다.

## 2025.05.03 (토)
**[지브리풍 이미지 생성 열풍, 생성형 AI 대중화로의 진입](https://n.news.naver.com/mnews/article/006/0000129781?sid=105)** ```AI``` ```Generative AI```

* 2025년 3월 말부터 유행한 **ChatGPT Image Generation 을 이용한 지브리풍 이미지 생성 열풍** 이 생성형 AI에 대한 국민들의 관심 및 이용률에 크게 기여했다.
  * 심지어는 생성형 AI의 답변을 **언론사 뉴스보다 신뢰한다는 응답** 까지 있었다.
* 한국언론진흥재단의 '미디어서베이' 결과에 따르면 다음과 같다.
  * 최근 1개월간 생성형 AI 사용 경험률 **57.2%**
    * 이 중 최근 1개월간 **처음** 생성형 AI를 사용해 본 사람은 무려 **37.0%** 
    * **2025년에 처음** 생성형 AI를 사용해 본 사람은 무려 **54.9%** 로 **절반 이상**
    * 지브리 스타일 등 사진 변환 경험률 **59.5%**
  * 생성형 AI 활용 목적은 **정보 검색 (81.0%) > 작문 (51.1%) > 이미지 생성 및 보정 (51.0%)** 순
  * 생성형 AI 사용에 대한 우려로, **스스로 생각하는 시간의 감소 (73.0%)** 등을 주로 우려하고 있음

## 2025.05.02 (금)
**[AI 분야 1.9조원 추경 의결](https://n.news.naver.com/mnews/article/032/0003367090?sid=105)** ```AI```

* 2025.05.01일 국회 본회의에서 **AI 분야 추경 예산으로 1.9조 원 가량 (1조 9067억 원) 이 의결** 되었다.
* 이 추경 예산으로 실행할 계획은 다음과 같다.
  * **AI 인프라 확대**
    * 첨단 GPU 1만 장 신규 확보 및 민간 첨단 GPU 임차 활용
    * 신경망처리장치 (NPU) 등 반도체 실증 예산 2배 이상 증액
  * AI 국가대표 팀이 글로벌 수준 AI 모델을 개발하는 **'월드 베스트 LLM' 프로젝트**
  * 최고 수준의 해외 AI 연구원 국내 유치 프로젝트
  * AI 혁신펀드를 통한 **AI 스타트업 지원** 의 정부 예산 증액

## 2025.05.01 (목)
**[카카오, 국내 최초 멀티모달 LLM 공개](https://n.news.naver.com/mnews/article/016/0002465776?sid=105)** ```AI``` ```Kakao AI``` ```Large Language Model```

* 2025.05.01일 카카오에서 국내 최초의 멀티모달 LLM인 **카나나-오 (Kanana-o)** 를 공개했다.
  * '국내 최초의 멀티모달 LLM'이라 함은, **텍스트, 이미지, 음성 등 다양한 형태의 데이터를 동시에 분석하는 모델** 을 의미한다.
* 카카오에서 개발한 모델은 다음과 같다.

| 모델                | 설명                                                                 |
|-------------------|--------------------------------------------------------------------|
| 카나나-오 (Kanana-o)  | 통합 멀티모달 LLM (국내 최초의 멀티모달 LLM)<br>- **카나나-브이, 카나나-에이 모델을 통합** 하여 개발 |
| 카나나-에이 (Kanana-a) | 오디오 언어 모델                                                          |
| 카나나-브이 (Kanana-v) | 이미지 데이터 특화 모델                                                      |

* 카카오에서 멀티모달 LLM 개발을 위해 사용한 기술은 **병합 학습 (동시에 여러 형태의 데이터를 학습)** 이다.
  * 이를 통해 다양한 형태의 정보를 텍스트와 연결하도록 **통합 훈련**
* 카나나-오의 성능 수준은 다음과 같다.
  * 한국어 및 영어의 경우, **글로벌 SOTA 모델 수준의 성능**
  * 한국어 및 영어에서의 **감성 인식** 의 경우, **감정 이해 및 소통이 가능한 AI임을 입증** 하는 수준의 성능을 보였다.
