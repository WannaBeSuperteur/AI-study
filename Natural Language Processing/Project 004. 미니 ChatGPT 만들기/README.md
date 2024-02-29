# NLP Project 004. 미니 ChatGPT 만들기
* Dataset: [Human Conversation training data](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data)
  * Human 1, Human 2 라는 두 사람이 대화를 나누는 것에 대한 학습 데이터이다.
  * 다운로드 버튼을 클릭하여 ```archive.zip``` 파일을 다운받은 후, 그 파일의 압축을 풀어서 나온 ```archive/human_chat.txt``` 파일이 학습 데이터이다.
* 수행 기간: 2024.02.25 ~ 2024.03.10 (15일)

## 파일 및 코드 설명
* ```tokenize_data.py``` : 학습 데이터에 대한 tokenize 진행
* ```generate_data.py``` : 모델 학습에 필요한 입력 데이터 및 출력 데이터로 구성된 데이터셋인 **토큰 예측 학습 데이터** 생성
  * 출력 파일 : ```train_data.csv``` (학습 데이터)
* ```embedding_helper.py``` : token의 one-hot encoding, dictionary에서의 index 값으로 변환 등 임베딩 관련 함수
* ```train_embedding_model.py``` : **임베딩 모델** 에 대한 학습 실시
  * 필요 파일 : ```train_data.csv```
  * 출력 모델 : ```embedding_model``` (임베딩 모델)
* ```train_latent_vector_model.py``` : **latent vector 모델** 에 대한 학습 실시
  * 필요 파일 및 모델 : ```train_data.csv```, ```embedding_model```
  * 출력 모델 : ```latent_vector_model``` (latent vector 모델)
* ```train_main_model.py``` : **메인 모델** 에 대한 학습 실시
  * 필요 파일 및 모델 : ```train_data.csv```, ```embedding_model```
    * 현재는 사용하지 않지만, 경우에 따라 ```latent_vector_model``` 모델 사용 가능
  * 출력 모델 : ```main_model``` (메인 모델)
* ```test.py``` : 학습으로 만들어진 모델 테스트
  * 필요 모델 : ```main_model```
* ```main.py``` : 전처리, 학습 실시, 테스트 실시의 모든 과정을 한번에 진행

## 데이터 전처리 및 생성 과정
* tokenize 방법
  * 문장 부호 및 단어 간 공백 단위로 tokenize (추후 변경 가능)
  * 말하는 사람이 한 사람에서 다른 사람으로 전환되는 부분은 특별한 token을 생성
    * 예: ```Person 1: Natural Language Processing gives machine a life. Do you agree?, Person 2: Yes, ChatGPT is the best example!``` 에 대해, token 구성은 ```[..., "life", ".", "Do", "you", "agree", "?", "<Person_Change>", "Yes", ",", "ChatGPT", "is", ...]``` 가 된다. 이때 ```<Person_Change>``` 라는 특별한 token이 삽입되었다.
* 데이터 생성 방법
  * 학습 데이터 전체를 처음부터 읽어 나가면서, **전체 학습 데이터의 첫 번째 token부터 ~ 마지막 token을 기준으로 16개 token 만큼 이전의 token까지** 의 각 token을 시작점으로,
    * 연속된 17개의 token을 추출
    * 17개의 token 중 첫 16개 token은 입력 데이터로, 마지막 1개 token은 출력 데이터로 지정
    * 첫 16개 token에 근거하여 마지막 1개 token을 예측하는 모델을 생성하도록 데이터 구성
  * 위와 같은 방법으로 구성한 데이터셋을 **토큰 예측 학습 데이터** 라 하자. 이 데이터셋에서, 첫 90%는 train data, 마지막 10%는 validation data
    * train, valid 데이터 구분은 데이터를 지정하거나 실제 Tensorflow를 이용하여 학습할 때 split_ratio 등을 이용하여 적용한다.
  * 위 데이터는 ```train_data.csv``` 파일에 저장한다.

## 머신러닝 모델 설명
* 각 token에 해당하는 word 를 저장하고 one-hot encoding 할 수 있는 **dictionary (=vocab)** 필요
* **메인 모델** (실질적으로 ChatGPT에서 답변을 출력하는 역할을 하는 NLP 모델)
  * 입력 : **토큰 예측 학습 데이터** 에서, 입력 데이터에 해당하는 16개의 token (각각 24-dim vector로 embedding) (+ latent vector (dimension = 16), **현재 latent vector는 사용하지 않음**)
    * latent vector는 **현재 사용하지 않으며,** 입력 데이터에 해당하는 16개의 token을 **latent vector 모델** 에 넣었을 때 생성되는 latent vector 임.
    * 실제 모델 테스트 시, 입력 데이터 중 16개의 token 이외의 latent vector **(현재 미사용)** 부분의 경우, 입력 데이터 token을 latent vector 모델에 넣어서 생성된 값 대신 원하는 값으로 **자유롭게 조작 가능**
  * 출력 : **토큰 예측 학습 데이터** 에서, 출력 데이터에 해당하는 (output embedding 과 가장 가까운) 1개의 token
    * output 은 embedding vector 크기만큼의 크기 (=24) 를 갖는 배열로, 출력값으로 가장 적절한 1개의 token을 예측
    * output 되는 token 은 vocab의 각 단어를 S-BERT 임베딩한 결과 벡터를 **임베딩 모델** 에 넣었을 때 출력되는 임베딩 중 output 과 가장 가까운 Euclidean 거리의 embedding 에 해당하는 단어
  * 모델 구조 : 입력 데이터에 해당하는 token 들을 **dictionary 를 이용하여 token id -> embedding (**임베딩 모델** 이용, for each token) -> concatenate -> Neural Network -> output** 으로 진행하여 출력
* **latent vector 모델** (Auto-Encoder 구조)
  * 입력 : **토큰 예측 학습 데이터** 에서, 입력 데이터에 해당하는 16개의 token의 **임베딩 모델** 에 의한 embedding 의 concatenation (단, 여기에 random noise 추가)
  * 출력 : random noise가 없는 원본 입력 데이터와 동일
  * 입력 데이터에 해당하는 token을 **dictionary 를 이용하여 embeddings (**임베딩 모델** 이용, 24 for each token) -> common NN -> concatenate (4-dim vector each for 16 tokens) -> Neural Network -> latent vector (16) -> Neural Network -> split (4-dim vector each for 16 tokens) -> inverse common NN -> embeddings (24 for each token)** 으로 진행하여 출력
    * common NN : **임베딩 모델** 에 의한 임베딩 (vector size 24) 을 입력받아서 vector size 4의 벡터를 출력하는, 모든 embedding에 대해 공통으로 사용되는 간단한 신경망 구조
    * inverse common NN : vector size 4의 벡터를 입력받아서 **임베딩 모델** 에 의한 임베딩 (vector size 24) 를 출력하는, 마찬가지로 모든 embedding에 대해 공통으로 사용되는 간단한 신경망 구조
  * 학습 목적 : 입력 데이터를 나타내는 latent vector 생성
    * 해당 latent vector는 메인 모델의 입력 데이터로 사용
    * 해당 latent vector에 랜덤한 noise를 추가하거나 조작하여 **언어를 생성하는 모델** 구현이 목표
* **임베딩 모델**
  * 입력 : 각 단어의 S-BERT 임베딩 (384) 중 처음 128개 (128) * ```일정 값``` + ```small random noise```
  * 출력 : 입력 데이터와 동일한 S-BERT 임베딩 (384) 중 처음 128개 (128) * ```일정 값```
  * 학습 데이터 : **토큰 예측 학습 데이터** 의 각 row 에서 2번째 단어를 입력, 이와 동일한 2번째 단어를 출력에 이용
  * 모델 구조 : **입력 -> Neural Network -> embedding layer (24) -> Neural Network -> 출력** (embedding layer 의 값이 word/token embedding 에 해당함)
* 학습 순서는 **임베딩 모델 -> latent vector 모델 -> 메인 모델**

### 아이디어
* **(최종 채택)** 생성형 언어 모델로 만들기 위해서, latent vector 사용 시의 **언어 생성 능력** 관점에서의 모델 성능이 좋지 않은 경우 output 과 동일한 shape 의 가중치 (weight) array를 0.5~1.5 등 적절한 특정 범위의 값들로 랜덤하게 초기화한 후, **output과 각 embedding에 대해 해당 array를 각각 dot product (output * weight, embedding * weight)** 해서 거리를 구하는 것은 어떨까?
* 가장 적절한 token을 예측할 때, **메인 모델** 의 출력 배열과 Euclidean Distance 기준으로 가장 가까운 embedding 의 단어 1개만을 hard하게 출력하는 대신, Euclidean Distance 가 일정 값 이하인 모든 임베딩에 대해, 그 거리에 반비례하는 확률로 해당 임베딩에 해당하는 단어를 확률적으로 출력하게 하면 어떨까?
  * 이렇게 하면 latent vector를 사용하지 않아도 되는데, 이것이 장단점이 있다.
    * 장점 : 모델을 **메인 모델** 만 사용해도 되기 때문에, 전체적인 프로젝트 구조가 간단해진다.
    * 단점 : latent vector를 조작하여 **특정한 어투 등을 반영하여** 문장을 생성하도록 할 수 없다.

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
|NLP-P4-master|||240225|240310|마스터 브랜치|
|NLP-P4-1|```done```|```feat```|240225|240225|학습 데이터 tokenize 진행|
|NLP-P4-2|```done```|```feat```|240226|240226|**토큰 예측 학습 데이터** 생성|
|NLP-P4-3|```done```|```feat```|240226|240229|**latent vector 모델** 구성 및 해당 모델의 학습 실시|
|NLP-P4-4|```done```|```feat```|240226|240226|**임베딩 모델** 구성 및 해당 모델의 학습 실시|
|NLP-P4-5|```ing```|```feat```|240229||**메인 모델** 구성 및 해당 모델의 학습 실시|
|NLP-P4-6||```feat```|||학습 모델 테스트|
|NLP-P4-7||```feat```|||전처리, 학습, 테스트의 모든 과정을 진행하는 ```main.py``` 파일 작성|
|NLP-P4-8||```feat```|||모델 정성평가용으로, 사용자가 입력하면 **메인 모델** 이 답변을 출력하는 부분 작성|
