# 자연어 처리 (NLP) 기초 - BERT, SBERT (Sentence BERT) 모델

2023.12.08 작성중

## BERT 모델이란?
[BERT 논문](https://arxiv.org/pdf/1810.04805.pdf)

**BERT (Bidirectional Encoder Representations from Transformers)** 는 Google AI의 언어 모델 연구진이 개발하여 2018년에 공개한 NLP 모델이다.

![BERT 모델의 학습](./images/BERT_1.PNG)

(출처: Jacob Devlin, Ming-Wei Chang et al, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)

위 그림은 BERT 모델의 Pre-training과 Fine-tuning의 학습을 나타낸 것이다.

![BERT 모델의 입력 표현](./images/BERT_2.PNG)

(출처: Jacob Devlin, Ming-Wei Chang et al, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)

위 그림은 BERT 모델의 입력에 대한 Token, Segment, Position 임베딩을 나타낸 것이다.

## SBERT 모델