# 자연어 처리 (NLP) 기초 - 토크나이저 (tokenizer)

## 토큰 (token) 이란?
**토큰 (token)** 이란 NLP 모델이 자연어 처리를 위해 사용하는 언어의 최소 단위 (형태소와 유사) 이며, 주어진 문장 등을 토큰으로 나누는 작업을 **토큰화 (tokenization)** 라고 한다.
* 형태소 기준으로 토큰화 시, 토큰의 개수가 단어의 개수보다 약간 많아진다. ChatGPT의 경우 750개의 단어 당 1,000개의 토큰으로 토큰화된다고 한다.

토큰화를 위해 다음과 같은 방법들을 생각해 볼 수 있다. 예시 문장은 **Natural Language Processing gives machine a life. Don't you agree?** 이다.

* 1. 단어 단위로 토큰화한다. **(단어 토큰화, Word Tokenization)**
  * 물음표와 마침표를 제외하고 예시 문장을 단어 단위로 토큰화하면 ```["Natural", "Language", "Processing", "gives", "machine", "a", "life", "Don't", "you", "agree"]``` 이다.
* 2. **Penn Treebank Tokenization**
  * 하이픈으로 구성된 단어는 유지하고, 아포스트로피를 고려하여 단어를 분리하는 토큰화 방법이다.
  * 예시 문장을 단어 단위로 토큰화하면 ```["Natural", "Language", "Processing", "gives", "machine", "a", "life", ".", "Do", "n't", "you", "agree", "?"]``` 이다.
* 3. **문자 기반 토큰화**
  * 텍스트를 단어나 형태소가 아닌 문자의 집합으로 간주하여 문자 단위로 나눈다.
  * 예시 문장의 경우 ```["N", "a", "t", ..., "r", "e", "e", "?"]``` 로 토큰화된다.
  * 일본어나 중국어 등 일부 언어를 제외하면, 직관적인 의미 파악이 어렵다.

영어의 경우 토큰화가 쉽지만, 한국어는 띄어쓰기가 잘 지켜지지 않는 등의 이슈가 있어서 토큰화가 비교적 어렵다.

## 품사 태깅 (PoS Tagging)
**품사 태깅 (Part-of-speech Tagging, PoS Tagging)** 이란, **표기는 같지만 품사에 따라 의미가 달라지는** 단어가 있는 문장에서 그 의미를 언어 모델이 파악할 수 있게 하거나, 이런 문장을 전처리하기 위한 목적으로 각 단어나 토큰의 품사를 표시하는 것이다.

## 토크나이저 (tokenizer) 란?
**토크나이저 (tokenizer)** 는 앞에서 제시한 것과 같은 방법들을 이용하여 토큰화를 하는 도구로, 사용자가 입력한 텍스트 문장을 NLP 모델이 처리할 수 있는 형태로 변환하기 위해 사용한다.
* 모델에 적합하면서도 간결한 표현을 찾는 것이 목표이다.