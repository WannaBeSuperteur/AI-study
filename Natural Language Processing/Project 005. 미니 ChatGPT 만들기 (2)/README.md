# NLP Project 005. 미니 ChatGPT 만들기 (2)
* Dataset: [Human Conversation training data](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data)
  * Human 1, Human 2 라는 두 사람이 대화를 나누는 것에 대한 학습 데이터이다.
  * 다운로드 버튼을 클릭하여 ```archive.zip``` 파일을 다운받은 후, 그 파일의 압축을 풀어서 나온 ```archive/human_chat.txt``` 파일이 학습 데이터이다.
* 수행 기간: 2024.03.01 ~ 2024.03.10 (10일)

## 파일 및 코드 설명
* ```tokenize_data.py``` : 학습 데이터에 대한 tokenize 진행
* ```generate_data.py``` : 모델 학습에 필요한 입력 데이터 및 출력 데이터로 구성된 데이터셋 **(모델의 학습 데이터)** 생성
  * 출력 파일 : ```train_data.csv``` (학습 데이터)
* ```embedding_helper.py``` : token의 one-hot encoding, dictionary에서의 index 값으로 변환 등 임베딩 관련 함수
* ```train.py``` : 모델에 대한 학습 실시
  * 필요 파일 및 모델 : ```train_data.csv```
  * 출력 파일 : ```embedding_result.csv``` (생성형 출력을 위한 token 별 embedding 결과)
  * 출력 모델 : ```mini_chatgpt_model``` (모델)
* ```test.py``` : 학습으로 만들어진 모델 테스트
  * 필요 파일 : ```embedding_result.csv``` (생성형 출력을 위한 token 별 embedding 결과)
  * 필요 모델 : ```mini_chatgpt_model```
* ```main.py``` : 전처리, 학습 실시, 테스트 실시의 모든 과정을 한번에 진행

## 데이터 전처리 및 생성 과정
* tokenize 방법
  * 문장 부호 및 단어 간 공백 단위로 tokenize
  * 말하는 사람이 한 사람에서 다른 사람으로 전환되는 부분은 특별한 token을 생성
    * 예: ```Person 1: Natural Language Processing gives machine a life. Do you agree?, Person 2: Yes, ChatGPT is the best example!``` 에 대해, token 구성은 ```[..., "life", ".", "Do", "you", "agree", "?", "<Person_Change>", "Yes", ",", "ChatGPT", "is", ...]``` 가 된다. 이때 ```<Person_Change>``` 라는 특별한 token이 삽입되었다.
* 데이터 생성 방법
  * 학습 데이터 전체를 처음부터 읽어 나가면서, **전체 학습 데이터의 첫 번째 token부터 ~ 마지막 token을 기준으로 24개 token 만큼 이전의 token까지** 의 각 token을 시작점으로,
    * 연속된 25개의 token을 추출
    * 25개의 token 중 첫 24개 token은 입력 데이터로, 마지막 1개 token은 출력 데이터로 지정
      * 첫 24개 token 중 발화자 변경에 따른 특별한 토큰인 ```<Person_Change>``` 가 있을 시, 마지막으로 등장하는 ```<Person_Change>``` 토큰 이전의 모든 토큰을 ```<Null>``` 토큰으로 변경한 데이터를 입력 데이터로 하여, 동일한 출력 데이터를 가지는 새로운 data row를 같이 추가한다.
    * 첫 24개 token에 근거하여 마지막 1개 token을 예측하는 모델을 생성하도록 데이터 구성
  * 위와 같은 방법으로 구성한 데이터셋에서, 첫 90%는 train data, 마지막 10%는 validation data
    * train, valid 데이터 구분은 데이터를 지정하거나 실제 Tensorflow를 이용하여 학습할 때 split_ratio 등을 이용하여 적용한다.
  * 위 데이터는 ```train_data.csv``` 파일에 저장한다.

## 머신러닝 모델 설명
* 각 token에 해당하는 word 를 저장하고 one-hot encoding 할 수 있는 **dictionary (=vocab)** 필요
* **모델 설명** (실질적으로 ChatGPT에서 답변을 출력하는 역할을 하는 NLP 모델)
  * 입력 : 학습 데이터에서, 입력 데이터에 해당하는 24개의 token의 ID
  * 출력 : 학습 데이터에서, 출력 데이터에 해당하는 1개의 token에 대해 그 ID를 이용한 one-hot vector (크기: vocab) 
    * 출력값으로 가장 적절한 1개의 token을 예측하며, 가장 큰 값에 해당하는 ID의 token을 최종 출력
    * **생성형 출력** 을 위해, 다음과 같은 행렬 및 연산을 이용
      * 모델 중 word embedding 하는 부분을 따로 떼어내서, 해당 부분에 각 token를 넣어서 embedding 된 결과를 저장한 **(vocab size x embedding dimension)** 차원의 행렬 **A** (```embedding_results.csv``` 파일로 저장)
      * embedding 의 각 원소에 대한 가중치를 담은 **(embedding dimension)** 차원의 randomly initialized 된 배열 **B**
      * 모델의 원래 출력값을 $O$라고 할 때, 다음과 같이 각 index $i$에 대해 변환된 출력값 $O'$ 를 이용 (단, $v = 0,1,...,V-1$, $d = 0,1,...,D-1$, $V$ = (vocab size), $D$ = (embedding dimension), $i$ 는 출력값 $O$ 의 해당 단어의 index)
      * $\displaystyle O'^{i} = O^i \times \Sigma_{d=0}^{D-1} (|A_{i,d}| \times B_{d}), i=0,1,...,V-1$ (단, $O^i, O'^i$ 는 각각 $O$, $O'$ 의 index $i$ 의 값)
    * **생성형 출력** 을 위해, 위 수식에 따라 변환된 출력값 $O'$ 에서 **값이 가장 큰 index에 대응되는 token** 이 아닌, **값이 일정 threshold (예: 0.05) 이상인 index의 모든 token을, 해당 값에 비례하여 확률적으로** 출력
      * 예를 들어, $O' = [0.02, 0.5, 0.25, 0.03, 0.01, 0.04, 0.15]$ (vocab size = 7) 이고 해당 threshold가 0.05일 때, 0.5에 해당하는 index의 token을 출력할 확률은 100% 가 아니라 **0.5 / (0.5 + 0.25 + 0.15) = 55.6%**  
  * 모델 구조 : 입력 데이터에 해당하는 token 들을 **token ID -> embedding -> LSTM -> Dense Layers -> output (with vocab size) -> softmax로 token 출력** 을 적용

## 실행 순서
```
python main.py
python test.py
```

## 성능지표 결과
* 성능 측정지표 : 정성 평가로 진행
  * ```test.py``` 파일 실행 시, 입력 문장을 받은 후, **메인 모델** 이 출력하는 답변을 확인할 수 있음
  * 이때, **메인 모델** 은 다음 token 예측을 15회 정도 또는 마침표 / ```<Person-Change>``` 토큰이 올 때까지 반복하여, 예측한 token을 모두 연결하여 완전한 문장을 출력하게 한다.

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|NLP-P5-master|||240301|240310|마스터 브랜치|
|NLP-P5-1|```done```|```fix```|240301|240301|tokenizer 개선|
|NLP-P5-2|```done```|```feat```|240301|240301|학습 데이터 생성|
|NLP-P5-3||```feat```|||모델 구성 및 해당 모델의 학습 실시|
|NLP-P5-4||```feat```|||전처리, 학습, 테스트의 모든 과정을 진행하는 ```main.py``` 파일 작성|
|NLP-P5-5||```feat```|||모델 정성평가용으로, 사용자가 입력하면 모델을 통해 답변을 출력하는 부분 작성|
