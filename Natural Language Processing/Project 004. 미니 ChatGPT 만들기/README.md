# NLP Project 004. 미니 ChatGPT 만들기
* Dataset: [Human Conversation training data](https://www.kaggle.com/datasets/projjal1/human-conversation-training-data)
  * Human 1, Human 2 라는 두 사람이 대화를 나누는 것에 대한 학습 데이터이다.
  * 다운로드 버튼을 클릭하여 ```archive.zip``` 파일을 다운받은 후, 그 파일의 압축을 풀어서 나온 ```archive/human_chat.txt``` 파일이 학습 데이터이다.
* 수행 기간: 2024.02.25 ~ 2024.03.10 (15일)

## 파일 및 코드 설명
* ```tokenize_data.py``` : 학습 데이터에 대한 tokenize 진행
* ```generate_data.py``` : 모델 학습에 필요한 입력 데이터 및 출력 데이터로 구성된 데이터셋인 **토큰 예측 학습 데이터** 생성
* ```embedding_helper.py``` : token의 one-hot encoding, dictionary에서의 index 값으로 변환 등 임베딩 관련 함수
* ```train_latent_vector_model.py``` : **latent vector 모델** 에 대한 학습 실시
* ```train_main_model.py``` : **메인 모델** 에 대한 학습 실시
* ```test.py``` : 학습으로 만들어진 모델 테스트
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

## 머신러닝 모델 설명
* 각 token에 해당하는 word 를 저장하고 one-hot encoding 할 수 있는 **dictionary (=vocab)** 필요
* **메인 모델** (실질적으로 ChatGPT에서 답변을 출력하는 역할을 하는 NLP 모델)
  * 입력 : **토큰 예측 학습 데이터** 에서, 입력 데이터에 해당하는 16개의 token + latent vector
    * latent vector는 입력 데이터에 해당하는 16개의 token을 **latent vector 모델** 에 넣었을 때 생성되는 latent vector 임.
  * 출력 : **토큰 예측 학습 데이터** 에서, 출력 데이터에 해당하는 1개의 token
  * 입력 데이터에 해당하는 token 들을 **dictionary 를 이용하여 one-hot encoding -> embedding -> concatenate -> Neural Network -> output** 으로 진행하여 출력
  * output 은 dictionary (vocab) 의 크기만큼의 크기를 갖는 배열로, 출력값으로 가장 적절한 1개의 token을 예측
* **latent vector 모델** (Auto-Encoder 구조)
  * 입력 : **토큰 예측 학습 데이터** 에서, 입력 데이터에 해당하는 16개의 token
  * 출력 : 입력과 동일
  * 입력 데이터에 해당하는 token을 **dictionary 를 이용하여 one-hot encoding -> concatenate -> Neural Network -> latent vector -> Neural Network -> split -> one-hot encoding** 으로 진행하여 출력
  * 학습 목적 : 입력 데이터를 나타내는 latent vector 생성
    * 해당 latent vector는 메인 모델의 입력 데이터로 사용
    * 해당 latent vector에 랜덤한 noise를 추가하거나 조작하여 **언어를 생성하는 모델** 구현이 목표
    * 가장 적절한 token을 예측할 때, **메인 모델** 의 출력 배열에서 가장 큰 값에 해당하는 index의 단어 1개만을 hard하게 출력하는 대신, 그 값이 일정 값 이상인 모든 index에 대해, 그 값의 크기에 비례하는 확률로 해당 index들에 해당하는 단어를 확률적으로 출력하게 하면 어떨까?
      * 예: dictionary (vocab) 의 size가 5라고 하자. 이때, 어떤 입력에 대한 **메인 모델** 의 출력 배열이 ```[0.03, 0.6, 0.2, 0.1, 0.07]``` 일 때, ```0.6```의 index 해당하는 단어만 출력하는 대신, ```0.1``` 이상인 모든 index (```0.6```, ```0.2```, ```0.1```) 에 대해 그 값에 비례해서 확률적으로 단어를 출력한다. 예를 들어 ```0.6```의 index에 해당하는 단어는 ```0.6 / (0.6 + 0.2 + 0.1) = 66.7%``` 의 확률로, ```0.2```의 index에 해당하는 단어는 ```0.2 / (0.6 + 0.2 + 0.1) = 22.2%``` 의 확률로 출력한다.
      * latent vector를 사용하지 않아도 되는데, 이것이 장단점이 있다.
        * 장점 : 모델을 **메인 모델** 만 사용해도 되기 때문에, 전체적인 프로젝트 구조가 간단해진다.
        * 단점 : latent vector를 조작하여 **특정한 어투 등을 반영하여** 문장을 생성하도록 할 수 없다.
* 학습 순서는 **latent vector 모델 -> 메인 모델**

## 실행 순서
```
python main.py
```

## 성능지표 결과
* 성능 측정지표 : 정성 평가로 진행
  * ```test.py``` 파일 실행 시, 입력 문장을 받은 후, **메인 모델** 이 출력하는 답변을 확인할 수 있음
  * 이때, **메인 모델** 은 다음 token 예측을 15회 정도 또는 마침표가 올 때까지 반복하여, 예측한 token을 모두 연결하여 완전한 문장을 출력하게 한다.

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|NLP-P4-master|||240225|240310|마스터 브랜치|
|NLP-P4-1||```feat```|||학습 데이터 tokenize 진행|
|NLP-P4-2||```feat```|||**토큰 예측 학습 데이터** 생성|
|NLP-P4-3||```feat```|||**latent vector 모델** 구성 및 해당 모델의 학습 실시|
|NLP-P4-4||```feat```|||**메인 모델** 구성 및 해당 모델의 학습 실시|
|NLP-P4-5||```feat```|||학습 모델 테스트|
|NLP-P4-6||```feat```|||전처리, 학습, 테스트의 모든 과정을 진행하는 ```main.py``` 파일 작성|
|NLP-P4-7||```feat```|||모델 정성평가용으로, 사용자가 입력하면 **메인 모델** 이 답변을 출력하는 부분 작성|