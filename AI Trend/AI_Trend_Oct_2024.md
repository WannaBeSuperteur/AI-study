## 2024.10.31 (목)
**['AI 아이폰'의 진화 계획](https://n.news.naver.com/mnews/article/003/0012872564?sid=105)** ```AI``` ```Apple AI```

* 애플이 iOS 18.1 운영 체제의 공식 배포를 시작으로 'AI 아이폰'의 개발에 본격 착수했다.
* 이러한 'AI 아이폰'의 신규 기능 개발 등 업데이트 계획은 다음과 같다.

|일정|OS 버전|상세 계획|
|---|---|---|
|2024.10.29|iOS 18.1, iPad OS 18.1, Mac OS 세쿼이아 15.1 업데이트|**애플 인텔리전스 기본 기능의 최초 포함** 버전 배포 완료<br>- AI가 글을 재작성, 요약 등을 수행하는 글쓰기 도구<br>- 시리의 대화 능력 향상<br>- 사진 앱에 AI 도입 등|
|2024.12월|iOS 18.2|애플 인텔리전스에 신규 기능 추가<br>- AI 기반 이미지 생성, 편집<br>- 글쓰기 도구 진화<br>- **글쓰기 도구, Siri, ChatGPT 간 통합**|
|2025.01 ~ 02월|iOS 18.3|기능 추가 또는 마이너 업데이트|
|?|iOS 18.4|시리의 개인화된 응답 향상 등 (예상)|

## 2024.10.30 (수)
**[생성형 AI를 활용한 신약 개발에 대해](https://n.news.naver.com/mnews/article/003/0012872628?sid=102)** ```AI``` ```Generative AI```

* 생성형 AI를 신약 개발에 활용할 때의 장점은 **텍스트, 이미지 등의 콘텐츠를 생성하여, 그것을 대규모 데이터의 학습 데이터와 유사한 형태의 학습 데이터로 사용** 할 수 있다는 것이다.
  * 즉, **머신러닝 모델의 학습 데이터 보충** 에 사용할 수 있다는 것이다.
  * 생성형 AI의 구체적인 장점은 다음과 같다.
    * 신약 개발 시, 특정 구조, 기능을 갖춘 단백질 구조 생성
    * 신약의 효능 및 안정성 예측
  * 국내의 경우 **생성형 AI 모델을 이용하여 3차원 단백질 결합을 효율적으로 분석/예측하는 '3bm GPT' 특허** 가 이미 존재한다.
* 그러나, 생성형 AI를 이용한 신약 개발에는 다음과 같은 과제가 따른다.
  * 복잡한 규제, 지적 재산권 문제, 데이터 프라이버시 등 **생성형 AI 자체의 널리 알려진 개인정보/보안 관련 위험**
  * 정확한 답을 찾지 못했을 때 발생하는 **환각 (Hallucination) 현상**
    * 이 때문에 부정확한 모델에 대한 필터링 기술이 필요하다. 

## 2024.10.29 (화)
**[AWS, 생성형 AI 로프트 투어 개최](https://n.news.naver.com/mnews/article/001/0015013120?sid=105)** ```AI``` ```Generative AI```

* AWS (아마존웹서비스) 에서 스타트업 및 AI 개발자 등에게 커뮤니티를 제공하기 위해 2024.10.30일부터 11.06일까지 **AWS 생성형 AI 로프트 투어** 를 개최한다.
  * 이 행사에 대해 AWS는 **스타트업의 혁신 가속화 기회** 가 될 것이라고 했다.
* 이 행사의 구체적인 내용은 다음과 같다.
  * 고객사 데모 등을 통해 AI 전문가와의 소통 & 최신 기술이 적용된 생성형 AI 모델 및 서비스에 대해 학습 기회 제공
  * 토스랩, 네오사피언스 등 **AWS의 고객사들이 직접 선보이는 생성형 AI 데모**
* 한편, AWS는 2016년 서울 서버 권역 출시 후 현재까지 지속적으로 클라우드 인프라 서비스를 제공 중이다. 

## 2024.10.28 (월)
**[서울대병원, AI 기반 수술 위험 예측하는 거대 언어 모델 개발](https://n.news.naver.com/mnews/article/018/0005870951?sid=102)** ```AI``` ```Large Language Model```

* 서울대학교병원 연구팀에서 마취 실시 전 평가 요약문을 바탕으로 **환자에 대한 수술을 실시했을 때 그 위험도를 평가하는 거대 언어 모델 (LLM) 을 자체 개발** 했다.
  * 서울대병원 마치통증의학과는 위와 같은 AI 모델을 개발하고, AUROC (예측 정확도) 0.915 라는 우수한 성능을 검증한 결과를 2024.10.28일 발표했다.
  * 이는 기존의 환자의 건강 상태를 1등급부터 6등급까지로 나타나는 **ASA-PS** 체계가 중증도 기준이 다소 주관점이고, 이 때문에 의료진 간 등급이 서로 불일치하게 평가된다는 것을 보완한 것이다.
  * 즉, **데이터 학습을 통해 환자들의 ASA-PS 등급을 자동 분류하는 LLM** 을 개발한 것이다.
* 상세한 학습 성능은 다음과 같다.
  * 환자 460명 데이터 기준, 모든 ASA-PS 등급에 대해 평균 예측 정확도 (AUROC) 91.5% 달성
  * 거대 언어 모델과 마취가 전문의의 분류 성적을 비교했을 때 다음과 같이 **거대 언어 모델의 성능이 약간 더 좋은 편** 이었다.
 
|분류 성적|특이도|정밀도|F1-Score|
|---|---|---|---|
|거대 언어 모델|90.1%|73.2%|71.6%|
|마취과 전문의|89.7%|71.5%|71.3%|
|편차|+0.4%p|+1.7%p|+0.3%p|

## 2024.10.27 (일)
**[언론 현장에서 활용되는 AI](https://n.news.naver.com/mnews/article/014/0005258832?sid=102)** ```AI``` ```Generative AI```

* AI에 대한 대중의 관심이 급증한 계기는 2016년 알파고, 2022년 ChatGPT의 등장이다.
  * 최근에는 2024년 노벨 과학상 (물리학상 & 화학상) 을 AI가 사실상 정복함으로써, AI가 인간의 영역을 대체하는 분위기이다.
* 해외에서는 최근 들어 다음과 같이 **기사 작성을 비롯한 새로운 서비스에 생성형 AI를 활용** 하고 있다.  
  * 그 구체적인 사례는 다음과 같다.
 
|사례|설명|
|---|---|
|자사 보도를 바탕으로 답변하는 각종 챗봇|- 영국 파이낸셜타임즈에서 출시한, LLM '클로드'를 활용한 챗봇 '애스크 FT' (Ask FT)<br>- 워싱턴소프트에서 개발한, 기후변화 관련 질문에 답변하는 'Climite Answers' 챗봇<br>- 국내 언론사들 중 일부도 AI 기업과 협력하여 자체 시스템 개발 중|

* 단, 현재까지 언론 부문에서는 **AI는 보조적인 역할을 하는** 수준에 그친다는 평가이다.
  * 특히, **생성형 AI는 오류 없이 안정적으로 기사를 작성할 수 없다** 는 AI 프로덕트 매니저의 발언도 있다. 

## 2024.10.26 (토)
**[생성형 AI로 랜섬웨어 제작, 첫 유죄 판결](https://n.news.naver.com/mnews/article/001/0015007876?sid=104)** ```AI``` ```Generative AI```

* 일본에서 **생성형 AI를 이용하여 랜섬웨어를 제작** 한 20대에게 징역 3년 + 집행유예 4년의 유죄 판결이 내려졌다.
  * 이러한 범행에 사용된 도구는 인터넷에 공개된 ChatGPT 비공식판 등을 포함한 복수의 생성형 AI였다.
  * 제작자는 IT 관련 전문 지식 없이도 AI의 도움으로 랜섬웨어를 쉽게 만들어냈다.
* 그가 랜섬웨어를 쉽게 만든 데에는 다음과 같은 점이 작용했다.
  * **생성형 AI가 범죄에 악용되지 못하도록 하는 일종의 자체 응답 제한을 우회**
    * 즉, 범죄 등에 **악용되지 않을 것으로 판단되는 질문을 반복** 하고, 그 결과물로 얻은 답변들을 조합하여 랜섬웨어 제작

## 2024.10.25 (금)
**['마테오', 제 1회 대한민국 AI국제영화제 대상 수상](https://n.news.naver.com/mnews/article/056/0011825596?sid=103)** ```AI``` ```Generative AI```

* 2024년 처음 개최된 '대한민국 AI 국제영화제' 는 **생성형 AI를 활용하여 창작한 영화** 만을 대상으로 하는 영화제이다. 이 영화제의 출품작들 중 '마테오'가 대상을 수상했다.
  * '마테오'는 성공을 위해 수단과 방법을 가리지 않는 주인공 '마테오'의 이야기를 담은 영화로, **마테오가 그의 삶에서 추구해야 하는 가치** 에 대해 논하는 영화이다.
  * 이 영화를 제작한 **마테오 AI 스튜디오의 문신우 씨** 는 이번 영화제가 단비 같은 기회가 되었다고 수상 소감을 밝혔다.
* 이 영화제에 출품된 2,067개의 작품 및 이들 중 본선에 오른 26개의 작품 중 다른 수상작들은 다음과 같다.  

|상격|작품|
|---|---|
|특별상|'리틀 마션즈: 사랑하는 나의 인간, 나의 뮤즈' (브라질)|
|4개 분야별 1등상|- 내러티브: '스토리'<br>- 다큐멘터리: '동굴의 신화, 연애'<br>- 예술 & 문화: '기억의 잔영'<br>- 자유 형식: '가을이 오면'|

## 2024.10.24 (목)
**[애플 인텔리전스, 다음 주 공식 출시](https://n.news.naver.com/mnews/article/003/0012861556?sid=105)** ```AI``` ```Apple AI```

* 애플의 AI 시스템인 **'애플 인텔리전스'가 애플이 다음 주에 배포할 새로운 OS인 iOS 18.1에 탑재** 된다고 한다.
  * iOS 18.1에는 애플 인텔리전스의 기능들 중 다음과 같은 기능이 일단 영어로만 제공된다.
    * 텍스트 관련 : 글쓰기 도구, 앱 알림 요약, 메시지 및 메일 등에 대한 답장 작성 지원 등
    * 이미지 및 생성형 AI 관련 : 사진 앱에서 원하지 않는 피사체 삭제 (클린 업), 사진 앱의 '추억 영화' 기능 등
  * 한편, **'시리'와 '챗GPT' 간의 결합** 역시 iOS 18.1 베타 버전에 포함된다.
    * 즉, 시리에 음성, 텍스트로 질문 시, **이용자 동의가 있으면 ChatGPT의 답변이 표시** 된다.
* 한편, 다음 업데이트인 iOS 18.2는 2024년 12월 정도로 예상된다.
* 또한, 애플 인텔리전스는 아이폰15 프로 계열, 아이폰16 시리즈 등 일부 기기에서만 사용 가능하다.

## 2024.10.23 (수)
**[예술인 1만 명, "AI가 생계 위협한다"](https://n.news.naver.com/mnews/article/032/0003327957?sid=104)** ```AI``` ```Generative AI```

* 노벨문학상 수상자를 비롯한 전 세계의 문화예술인들이 **문화예술 작품을 생성형 AI의 학습 데이터로 사용하는 데 반대** 하는 성명에 동의했다.
  * 이 성명은 창작자 지원 비영리 단체의 대표인 작곡가 '에드 뉴런 렉스'가 주도했으며, 현재까지 1만 명이 넘는 사람들이 서명했다.
    * 그는 'Stable Diffusion'이라는 이미지 생성 AI를 개발한 스타트업 '스태빌리티AI'에서 일한 경험이 있다.
  * 이 성명의 내용은 **생성형 AI 학습에 창작물을 무단 이용하는 것은 해당 작품 제작자의 생계를 부당하게 위협하는 것** 이라는 것이다.
* 한편, 미국에서는 여러 작가들이 AI 관련 기업들에 대해 저작권 관련 소송을 제기한 상태이다.

## 2024.10.22 (화)
**[카카오, 신규 AI 서비스 '카나나' 공개](https://n.news.naver.com/mnews/article/001/0014998330?sid=105)** ```AI``` ```Kakao AI```

* 카카오는 'if Kakao AI 2024'의 첫날인 2024.10.22일 **자사의 신규 AI 서비스인 '카나나' 를 공개** 했다.
  * 카나나는 일반적인 AI 에이전트를 넘어, **사용자의 친구가 되어 주는 'AI 메이트'를 지향** 한다는 점이 특징이다.
  * 이를 위해 **대화 맥락의 주요 정보를 기억** 한다는 특징을 가지고 있다.
* 카나나의 구성은 다음과 같다.

|구성|설명|
|---|---|
|나나 (nana)|개인 메이트로, **이용자와의 1:1 대화 및 그룹 대화** 를 기억<br>- 예를 들어, 그룹 대화에서 이야기된 각종 일정 등을 기억하여 메시지로 안내|
|카나 (kana)|**그룹 대화 내용만 기억** 하고, 이를 통해 사용자에게 도움<br>- 예를 들어, AI 관련 스터디에서 읽은 논문에 대한 퀴즈 출제 및 채점 가능<br>- '귓속말' 기능을 통해, 그룹 대화에 뒤늦게 참여한 사용자에게 그간의 대화를 요약|

* 카카오 측은 카나나는 카카오톡과 별개의 앱으로 출시할 예정이라고 했다.

## 2024.10.21 (월)
**[메타, 감정을 읽는 AI '스피릿LM' 공개](https://n.news.naver.com/mnews/article/092/0002349209?sid=105)** ```AI``` ```Meta AI``` ```Language Model```

* 페이스북과 인스타그램의 모회사이자 거대 언어 모델 (LLM) LLAMA 시리즈를 개발하여 오픈소스로 공개한 메타 (Meta) 가 **감정을 읽는 멀티모달 언어 모델인 '메타 스피릿 LM' 을 출시** 했다.
  * [Inference Code Github Link](https://github.com/facebookresearch/spiritlm)
* 메타 스피릿 LM의 특징은 다음과 같다.
  * 음성과 텍스트 형태의 데이터를 이용하는 멀티모달 모델 
  * **음성에 담긴 화자의 감정 상태를 분석** 하고 이를 모델의 출력에 반영 가능, 이를 통해 보다 자연스러운 의사소통 가능
  * **같은 의미를 갖는 텍스트와 음성을 함께 학습** 시키는 방법을 통해, 텍스트와 음성의 2가지 형태 데이터 간 상호 작용까지 학습
    * 이를 통해 **이들 두 형태 사이에서 소통이 가능** 한 언어 처리 능력 강화 
* 스피릿LM의 모델 규모는 다음과 같다.
  * 약 70억 개의 파라미터 (매개변수)
  * 스피릿LM 베이스/익스프레시브의 2가지 모델 제공

## 2024.10.20 (일)
**[네이버, 언어 모델 기술력 세계 학회 인정](https://www.ebn.co.kr/news/articleView.html?idxno=1640522)** ```AI``` ```Natural Language Processing``` ```Large Language Model``` ```Generative AI``` ```Naver AI```

* 2024.10.18일 네이버에 따르면 네이버의 검색 기술 관련 논문이 **EMNLP 2024 에 채택** 되었다고 한다.
  * EMNLP 는 자연어 처리 학회 중에서 세계 최고 수준의 권위를 자랑하는 곳이다.
  * EMNLP 2024는 2024.11.12일부터 11.16일까지 진행될 예정이며, 네이버는 이 자리에서 자사의 논문을 발표한다.
* EMNLP 2024에 채택된 네이버의 논문은 **자사의 생성형 AI를 이용한 검색 서비스인 큐(CUE:) 의 알고리즘** 에 대한 것이다.
  * 해당 알고리즘의 특징은 다음과 같다.
    * 모듈식 접근법으로, **소규모 언어 모델 (SLM)** 을 사용
    * 이를 통해 유해 질의 탐지, 적절한 답변 제공 가능 
* 이 외에도 **문서의 검색 노출 순위를 매기는 거대 언어 모델의 '지식'을 sLLM 에 이식하고, 이를 통해 실제 검색 서비스에 적용** 하는 방법에 대한 논문도 채택되었다.
  * 해당 방법의 의의는 **사용자가 필요로 하는 검색 결과의 실시간 제공 (속도 저하 없이)** 이다.

## 2024.10.19 (토)
**[AI 창작물에는 "AI 활용" 표시 의무화해야](https://n.news.naver.com/mnews/article/022/0003978254?sid=102)** ```AI``` ```Generative AI```

* 2024.10.19일 국회입법조사처에 따르면 입법처 소식지인 '이슈와 논점'에서 **AI가 창작한 창작물에는 그 창작에 AI를 활용했다는 것을 의무적으로 표시** 해야 한다고 주장했다.
  * 그 이유는 피해자 신고 또는 규제 기관의 모니터링만으로는 한계가 있기 때문이다.
  * 단, **모든 창작물** 을 대상으로 할지, 아니면 딥페이크 우려가 있는 창작물만을 대상으로 할지는 고민이 필요하다. 
    * 단, 법률적 혼란 우려를 최소화하기 위해 **모든 AI 창작물을 대상으로 하는 것이 좋다** 고 말했다.
* 한편, 국회에서 발의된 AI 관련 법률은 2024년 9월 말 기준 11건이며, 이 중 5건은 생성형 AI 콘텐츠에 대해 AI로 창작했다는 것을 표시하는 규정과 관련 있다.
* 또한, 유럽연합에서는 AI 생성 콘텐츠에 대해 이 사실을 판독할 수 있게 의무적으로 표시하도록 하고 있다.

## 2024.10.18 (금)
**[LG U+, '서울디자인 2024' 자체 AI 기술 '익시' 체험존](https://n.news.naver.com/mnews/article/421/0007852545?sid=105)** ```AI``` ```Generative AI```

* '서울 디자인 2024' 축제에 LG유플러스가 참여하여 **자사의 AI 기술인 익시 (ixi) 를 체험할 수 있는 공간** 을 제공한다.
* 체험존은 다음과 같이 구성된다.

|구성|설명|
|---|---|
|익시 포토부스|사진 1장을 촬영하면 이를 컨셉에 맞게 AI 프로필로 변환<br>- 사진을 촬영하면 SM엔터테인먼트의 가상 인간 나이비스 (naevis) 와 관련된 컨셉의 AI 프로필 사진 체험 가능|
|익시 프롬프트|생성형 AI 프롬프트를 이용하여 나만의 스마트폰 배경화면 생성<br>- 미래 도시 컨셉의 배경화면을 생성하는 AI 프롬프트 캠페인 진행|

* 한편, LG유플러스는 나이비스의 소속사 SM엔터테인먼트와 2024.09월 AI 콘텐츠 개발 관련 협업 제휴를 맺었다.

## 2024.10.17 (목)
**[생성형 AI 다음은 '행동형 AI'](https://n.news.naver.com/mnews/article/015/0005045541?sid=101)** ```AI```

* 2024.10.17일, 국내 AI 스타트업 생태계를 조망하는 행사인 **코리아 프라미싱 AI 스타트업 2024 (KPAS 2024)** 가 열렸는데, 여기서 AI 전문가들의 발언 중 **행동형 AI에서 혁신이 일어날 것** 이라는 내용이 있다.
  * '행동형 AI'는 기존의 '판단형 AI' 및 '생성형 AI'에 이은 그 다음 단계라고 할 수 있다.
  * 행동형 AI에 대해 발언한 전문가는 **텍스트, 이미지를 넘어 촉각, 후각, 미각 데이터를 AI가 학습하면 큰 변화가 일어날 것** 이라고 했다.
* 이 외에도 이 행사에서는 다음과 같은 것들이 논의되었다.
  * AI 사업화 영역 (각 분야에 특화된 AI 솔루션의 필요성)
  * 대한민국의 AI 역량을 분석한 결과, 기술 개발은 우수하지만 사업화 및 인재 확보 영역이 취약함   
  * **한국지능정보사회진흥원 (NIA)** 의 AI 융합 본부장은 **기술 개발보다 AI 활용에 주목** 해야 한다고 발언했다.

## 2024.10.16 (수)
**[유발 하라리 "AI는 사람인 척 상호작용 못하게 해야"](https://n.news.naver.com/mnews/article/028/0002711614?sid=103)** ```AI```

* 히브리대 역사학 교수 유발 하라리가 새로운 책 '넥서스'를 출간하면서 연 기자간담회에서 **AI는 '행위자'이기 때문에 인류에게 전에 없었던 문제를 야기할 수 있다** 고 주장했다.
  * 그는 **AI는 삶의 변화를 가속화하며, 생명체의 주기를 따르는 사람이 항상 켜져 있는 AI의 속도를 맞추려면 결국 무너질 수밖에 없다** 고 주장했다.
* 기자간담회에서 유발 하라리의 주요 발언들은 다음과 같다.
  * AI를 학습시키기 위해 사용하는 모든 데이터에는 편향이 있지만, 노력에 따라 편향에서 벗어나게 할 수 있다.
  * 국제적 긴장, 전쟁 등의 문제 상황에서 AI가 인류를 통제한다면 이를 막을 수는 없다.
  * 미래 사회에는 **AI가 사회의 변화를 가속** 시키기 때문에, **계속 새로 배우고 변화하는 능력** 이 중요하다.

## 2024.10.15 (화)
**[Adobe, 동영상 생성 AI 출시](https://n.news.naver.com/mnews/article/001/0014982841?sid=104)** ```AI``` ```Generative AI```

* 포토샵을 개발한 Adobe에서 2024.10.14일 **동영상을 생성하는 AI 모델인 파이어플라이 비디오 (Firefly Video) 을 출시** 한다고 밝혔다.
  * 이는 기존에 출시된 동영상 생성 AI인 OpenAI의 'Sora' (2024.02), 구글의 '비오' (2024.05), 메타의 '무비 젠' (2024.10) 과의 경쟁을 의미한다.
* '파이어플라이 비디오'의 특징은 다음과 같다.
  * 영상 편집 가능
  * Adobe 의 프리미어 프로 (Premiere Pro) 와 연동하여 '생성형 확장' 기능을 통한 영상 추가
    * 영상에서 **빠진 부분이나 뒷부분 등을 AI가 인식** 하여 동영상 생성
  * 안전한 상업적 사용 가능 
  * 기존의 'Sora', '무비 젠' 등과 달리 빅테크 최초로 **공개 시험 버전** 으로 출시

## 2024.10.14 (월)
**[뇌졸중 진단, AI가 석학을 이겼다](https://n.news.naver.com/mnews/article/346/0000081449?sid=004)** ```AI```

* 환자의 영상을 보고 뇌졸중 예후를 판단하는 대회에서 **뇌졸중 석학과 AI가 대결한 결과, AI 72점, 뇌졸중 석학 50점의 평균 점수로 AI가 승리** 했다.
  * 해당 대결이 펼쳐진 세미나는 2024.10.11일 개최된 **제1회 뇌졸중 AI 국제 검증 세미나** 이다.
* 이 대결에서는 MR 영상 이미지만을, 즉 **최소한의 시각 정보만을 이용** 하여 환자의 예후를 예측했다.
* 위 결과 이외에도 대회의 상세 결과는 다음과 같다.

|결과 항목|상세 결과|
|---|---|
|40 케이스 진단 속도|- AI가 월등히 뛰어남<br>- AI 12분 4초, 석학 평균 45분 43초 소요|
|전체 1위|AI가 아닌 해외 출신 석학|

* 이는 뇌졸중의 경우 **응급 환자의 경우 시술까지 걸리는 시간에 따라 환자의 예후가 달라지는** 만큼, 뇌졸중의 예후 파악에 있어서 **AI 솔루션을 활용하는 것이 좋은 방안** 이 될 수 있음을 의미한다.

## 2024.10.13 (일)
**[전 세계 100대 유니콘 중 21곳은 AI 스타트업](https://n.news.naver.com/mnews/article/293/0000059268?sid=105)** ```AI```

* 전 세계의 유니콘 스타트업 중 상위 100곳에 대해 조사한 결과, **이들 중 21곳이 AI 관련 스타트업** 이었다.
  * 국가별 비중은 **미국 18개사**, 중국 2개사, 오스트리아 1개사였고 대한민국 기업은 없었다.
  * 산업별 분포는 **엔터프라이즈 기술 15개사**, 제조 3개사, 미디어/엔터테인먼트 2개사, 헬스케어 1개사로 나타났다.
  * 세부 분야로는 **SAAS/데이터 솔루션 각각 5개사**, 거대 언어 모델 (LLM)/모빌리티/콘텐츠 각각 3개사, 금융/의료 각 1개사로 나타났다.
* 위와 같이 **AI 스타트업의 비중이 높다** 는 결과는 **AI가 혁신 생태계에 핵심적인 역할을 하고 있음** 을 나타낸다.
* 한편, 국내의 경우 AI 제도/규제 측면에서 미흡한 점이 있지만, AI 스타트업에 대한 투자가 이어지고 있으며 한국이 주요 산업에 대한 AI 적용 역량이 뛰어나다고 평가된 만큼 **대한민국의 AI 관련 잠재력도 충분** 하다고 할 수 있다. 

## 2024.10.12 (토)
**[AI 파마 코리아 컨퍼런스](https://n.news.naver.com/mnews/article/003/0012831480?sid=102)** ```AI```

* AI를 이용한 신약 개발 전략에 대한 정보를 공유하는 컨퍼런스인 **제 7회 AI 파마 코리아 컨퍼런스** 가 한국제약바이오협회와 한국보건산업진흥원의 주관으로 2024.10.31일 개최될 예정이다.
  * 이 컨퍼런스의 주제는 **AI 기술 대격변의 시대: 신약 개발의 혁신** 이다.
* 컨퍼런스의 상세한 내용은 다음과 같다.
  * 신약 개발에 AI를 적용하기 위한 4가지 전략
  * 신약 개발 분야에의 AI 기술 적용의 현재와 미래  
  * 데이터 큐레이팅을 이용한 AI 신약 개발
  * 이 외의 다양한 AI 정밀 의학, 신약 개발, AI 관련 정책 등에 대한 내용

## 2024.10.11 (금)
**[LLAMA 파생 모델 6만 개 + 메타, AGI 오픈소스 공개](https://n.news.naver.com/mnews/article/032/0003325461?sid=105)** ```AI``` ```Meta AI``` ```LLAMA``` ```Large Language Model``` ```AGI```

* 빅테크 기업 메타 (Meta) 에서 공개한 거대 언어 모델 LLAMA 를 이용하여 개발된 **파생 모델이 6.5만 개** 에 이른다.
  * 오픈소스 모델인 LLAMA 의 다운로드 횟수는 **4억 회** 에 이른다. 
  * 파생 모델 개발을 위해, 개발자는 공개된 LLAMA 모델을 추가 학습 (fine-tuning) 을 통해서 원하는 모델을 만들 수 있다. 
* 한편, 메타는 **인공 일반 지능 (AGI) 의 오픈소스를 공개** 한다는 방침을 발표했다.
  * 메타 측은 **메타의 장기적 목표는 AGI 구축 및 이에 대한 오픈소스 공개이고, 이를 통해 모든 사람이 AI의 혜택을 누리는 것** 이라고 밝혓다.
  * AGI는 '적어도 인간의 수준을 갖춘 AI'를 말하지만 현재까지 통일된 정의는 없다.
  * 한편, 메타 측은 AGI의 정의를 **커스터마이징이 불필요한 수준의 AI** 라고 한다면, 그 단계에 이르는 것은 아직 멀었다고 생각한다고 밝혔다.

## 2024.10.10 (목)
**[2024 노벨 화학상, AI 관련 인물들이 수상](https://n.news.naver.com/mnews/article/011/0004400955?sid=105)** ```AI```

* 2024년에는 노벨 물리학상 (AI 대부 '존 홉필드' 등) 에 이어 노벨 화학상까지 AI 관련 인물들이 수상한다.
* 2024년 노벨 화학상 수상자는 다음과 같다.
  * **데미스 허사비스 구글 CEO** (바둑 AI '알파고'를 개발한 구글 딥마인드 연구진의 리더)
  * **존 점퍼** (구글 딥마인드 수석 연구원)
  * **데이비드 베이커** (미국 워싱턴대 교수)
* 각 수상자들의 업적은 다음과 같다.

|수상자|업적|
|---|---|
|데미스 허사비스 & 존 점퍼|- 데미스 허사비스는 알파고 개발 연구진 리더<br>- 단백질 구조 분석 및 약물과의 상호작용 예측 AI 모델 '알파폴드' 개발, 빠른 단백질 분석에 기여|
|데이비드 베이커|- 2021년 로제타폴드 (알파폴드의 경쟁 AI 모델) 개발을 통해 '사이언스' 학술지에서 연구 성과 인정<br>- 로제타폴드는 약물을 직접 설계하는 기능까지 갖춘 AI|

## 2024.10.09 (수)
**[한글, 인공지능 시대에 잘 맞는 문자](https://n.news.naver.com/mnews/article/214/0001378898?sid=100)** ```인공지능``` ```자연어 처리```

* 한덕수 국무총리가 2024.10.09일 (한글날) 열린 한글날 경축식에서 **한글은 인공지능 시대에 잘 맞는 문자** 라고 평가했다.
  * 이와 함께 한국어 인공지능 관련 산업에 대한 지원 의지를 밝혔다.
  * 또한 인공지능 시장에서의 한국어 활용도 향상을 위해 **한국어 말뭉치 데이터 및 한국어에 능숙한 인공지능의 개발을 지원** 하겠다고 발언했다.
* [기사](https://n.news.naver.com/mnews/article/088/0000908586?sid=105) 에 따르면, 인공지능이 한국어를 학습하는 것과 관련하여 한국어는 다음과 같은 특징을 갖는다.
  * 다른 언어와 마찬가지로 글자를 토큰으로 나누어 그 패턴을 인식
  * **한글이 갖는 영어와의 어순 차이, 화자의 섬세한 맥락 표현력** 때문에, **다양한 형태의 패턴을 갖춘 대규모 데이터** 를 학습 데이터로 구축해야 한다.

## 2024.10.08 (화)
**[머신러닝 기초 확립 과학자, 노벨 물리학상 수상](https://n.news.naver.com/mnews/article/001/0014971605?sid=104)** ```AI```

* 2024년 노벨 물리학상은 **딥 러닝의 기반이 되는 인공신경망 기술을 이용한 머신러닝의 기반** 을 발견 및 발명한 과학자인 **존 홉필드, 제프리 힌턴** 이 수상하게 되었다.
  * 존 홉필드는 인공신경망의 물리적 모델인 **홉필드 네트워크 (Hopfield Network)** 를 발명했다.
* 이들 과학자에게는 약 13억 4천만 원 상당의 상금이 수여된다.
* 한편, 노벨상 시상식은 오는 12월 10일 열릴 예정이다.

## 2024.10.07 (월)
**[AI국제영화제 본선 진출 26편 확정](https://m.entertain.naver.com/article/001/0014968204)** ```AI``` ```Generative AI```

* 경기도에서 주최하는 '제1회 대한민국 AI 국제영화제'의 본선 진출작 26개 작품이 2024.10.07일 확정되었다.
  * 이 영화제는 **생성형 AI를 이용해 생성한 영화만을 대상으로 한** 영화제이다.
* 본선에 진출한 26개 작품은 국내 16편, 해외 10편으로 구성되어 있다.
* 시상은 2024.10.25일 킨텍스에서 열리며, 총 상금은 8100만 원이다.
* 한편, 심사위원장인 전찬일 영화평론가에 따르면 **최근 몇 달 동안 AI를 활용한 영상의 품질이 급상승했으며, 이는 스토리 개연성까지 갖춘 완성도 높은 영화의 탄생 가능성을 암시** 하는 것이라고 할 수 있다.

## 2024.10.06 (일)
**[구글, 동영상 내용 음성 검색 기능 출시](https://n.news.naver.com/mnews/article/001/0014963557?sid=104)** ```AI```

* 구글은 2024.10.03일 동영상 검색에서 **동영상의 내용에 대해 음성 검색을 통해 답을 알 수 있는** 기능을 출시한다고 했다.
  * 이는 텍스트 입력, 영상 속의 정지된 이미지 (사진) 에 대한 검색보다 한 단계 진화하여 **동영상의 내용** 에 대한 검색을 지원한다는 것을 의미한다.
  * 이는 구글의 AI 서비스인 **구글 렌즈** 를 이용한 것이다.
* 예를 들어 물고기 무리가 헤엄치는 장면을 동영상으로 찍은 후 물고기가 무리지어 헤엄치는 이유를 **음성으로** 질문하면, 관련 정보가 답으로 나오는 것이다. 
* 한편, 이 기능은 '구글 검색에 위협이 된다'는 이야기가 있었던 OpenAI의 자체 검색 엔진인 '서치GPT' 이후 약 2개월 만에 출시되었다.

## 2024.10.05 (토)
**[메타, 동영상 생성 AI '무비 젠' 출시](https://n.news.naver.com/mnews/article/001/0014965259?sid=104)** ```AI``` ```Generative AI```

* 2024.10.04일 메타에서 **동영상 생성 AI인 '무비 젠' (Movie Gen)** 을 공개했다.
  * 이를 통해 기존에 공개된 동영상 생성 AI인 OpenAI의 소라 (Sora), 구글의 비오 (Veo) 와의 경쟁 구도가 형성될 것으로 보인다.
  * 이 중 OpenAI의 Sora는 2024년 말까지 일반인이 사용 가능하도록 출시될 것으로 보인다. 
* 무비 젠의 특징은 다음과 같다.
  * 텍스트 입력을 통해 최대 16초 길이의 동영상 생성
  * **기존 동영상 편집, 오디오 생성, 사진의 인물이 등장하는 영상** 생성 가능 
* 무비 젠의 출시 계획은 다음과 같다.
  * Meta 내부 직원 중 일부, 영화 제작자 등 소수에게 우선 출시
  * 2025년 중 인스타그램 등 자사 앱에 탑재 예정 

## 2024.10.04 (금)
**[삼성 개발자 컨퍼런스 2024 - 모두를 위한 AI](https://n.news.naver.com/article/082/0001291272?sid=101)** ```AI```

* 2024.10.03일 삼성 개발자 컨퍼런스 2024가 **모두를 위한 AI - 10년의 개방적 혁신과 미래** 라는 주제로 열렸다.
  * 이 행사에서 삼성전자는 각 제품별로 특성에 맞는 AI를 적용하며, 이 제품 간 연결을 통해 **개인화된 AI를 구현** 할 예정이라고 했다.
* 이 행사에서 삼성전자는 위 내용 외에도 다음과 같은 AI 전략을 공유했다.
  * **말하는 사람이 누구이고, 어디에 있는지** 까지 고려한 개인화된 AI
    * 이를 통해 보안 향상 (예: 가족 구성원의 음성만을 인식)
  * AI 기술의 B2B 거래 확대

## 2024.10.03 (목)
**["AI가 내 일자리 위협" 불안, 한국인이 최상위](https://n.news.naver.com/mnews/article/032/0003324121?sid=105)** ```AI``` ```Generative AI```

* 생성형 AI 등 AI 관련 기술이 자신의 업무 능력 (일자리) 을 위협할지도 모른다는 우려 수준은 **대한민국이 다른 나라와 비교했을 때 높은 수준** 이라고 나타났다.
  * 한국보건사회연구원에서 발표한 '디지털 전환과 AI 기술에 대한 인식과 태도에 대한 10개국 비교' 보고서에서 대한민국은 해당 불안이 10개국 중 2위로 나타났다.
  * 해당 불안에 "동의한다" 또는 "매우 동의한다" 의 비율은 대한민국이 **이탈리아의 39.1% 에 이어서 35.4% 로 2위** 이다.
* 한편, 해당 조사에서 다룬 다른 내용들은 다음과 같다.
  * 디지털 기술 숙련도 : 대한민국 56.9% 로 54.9% 를 기록한 폴란드와 함께 하위권
  * 생성형 AI를 일터에서 사용하는 비율은 대한민국이 높은 편
  * "생성형 AI 개발에 대한 기관의 감시/규제 필요" 에는 10개국 모두에서 절반 이상 동의

## 2024.10.02 (수)
**[AWS, 생성형 AI '아마존 베드록' 출시](https://n.news.naver.com/mnews/article/031/0000873498?sid=105)** ```AI``` ```Generative AI```

* 아마존 AWS (Amazon AWS) 가 **생성형 AI 앱 개발을 위한 자동화 서비스인 아마존 베드록 (Amazon Bedrock)** 을 2024.10.02일 국내 정식 출시했다.
  * 이를 통해 **국내 고객의 생성형 AI 앱 개발 및 이를 통한 혁신을 지원** 함은 물론, 해당 앱을 실행 및 저장할 수 있는 위치를 더 다양하게 선택할 수 있다. 
* 아마존 베드록은 다음과 같은 기능을 지원한다.
  * 특정 목적에 적합한 거대 언어 모델 (LLM) 및 고성능 모델 탐색 및 접근
  * 생성형 AI를 적용한 앱의 구축 및 확장  
  * 보안, 개인정보 보호, 모델 커스터마이징 등의 기능
* 이날부터 국내 고객이 이용할 수 있는 최신 모델은 다음과 같다.

|기업/기관|모델|
|---|---|
|아마존 타이탄 (Amazon Titan)|Text Embeddings v2|
|앤스로픽 (Anthropic)|Claude 3.5 Sonnet, Claude 3 Haiku|

## 2024.10.01 (화)
**[AI가 정복하고 있는 새로운 영역들](https://n.news.naver.com/mnews/article/050/0000080380?sid=101)** ```AI```

* 비즈니스 측면에서, AI는 시간, 예산, 인력 등 **자원에 대한 투자 대비 최대 효과** 를 내는 일종의 '경쟁력'이다.
* AI를 적용하여 문제를 해결할 만한 분야는 다음과 같다.

|영역|설명|
|---|---|
|재산권 보호|**위조 상품 및 라이선스 부재 상품** 탐지 AI '마크비전' (MarqVision)<br>- AI를 통해 이미지 인식 및 맥락 분석<br>- 판매자 데이터 및 제품 데이터가 많을수록 더 정확한 위조 제품 구분 가능|
|로고 생성|- MS 빙 크리에이터에 프롬프트를 입력하면 'fine tuning' 식으로 로고 생성 가능|
|동영상 생성|- AI 서비스 'Gen2'를 통해 1장의 사진만으로 광고 등 영상 생성 가능|
|일상 생활|- 웹엑스의 AI 어시스턴트를 통해 읽지 않은 콘텐츠 요약<br>- **구글의 생성형 AI '제미나이'** 를 통해 다양한 상황에서의 길 안내 가능|