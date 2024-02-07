# NLP Project 002. 이모지 데이터 유사도 검증
* Dataset: [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
  * 문장과 이모지 표정이 ```;``` 로 구분된 텍스트의 집합
* 수행 기간: 2024.02.07 ~ 02.11 (5일)

## 파일 및 코드 설명
* ```train.txt``` (데이터셋을 다운받아야 함) : 학습 데이터 (```val.txt```, ```test.txt``` 는 사용하지 않음)

## 작업 설명
* 각 문장을 Sentence-BERT 를 이용하여 임베딩한다.
* 임베딩된 각 문장 중 랜덤으로 2개를 추출하는 시행을 N회 반복한다.
  * 이때, 각 문장 간 Sentence-BERT 임베딩의 코사인 유사도와, 해당 문장에 대응되는 이모지가 일치하는지의 여부를 확인한다.
* 두 문장의 이모지가 각각 ```이모지A```, ```이모지B``` (단, ```이모지A```와 ```이모지B```는 서로 다름) 인 모든 경우에 대해, 해당 두 문장의 Sentence-BERT 임베딩 결과의 코사인 유사도에 대한 평균값을 구한다.
  * 이를 통해 각 이모지가 Sentence-BERT 임베딩 기준 얼마나 의미적으로 서로 유사한지 판단한다.

## 실험 결과


## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|NLP-P2-master|||240207||마스터 브랜치|
|NLP-P2-1||```feat```|240207||데이터 읽기|
|NLP-P2-2||```feat```|||텍스트 임베딩하기|
|NLP-P2-3||```feat```|||임베딩된 문장 2개 추출 시행 반복 및 결과 저장|
|NLP-P2-4||```feat```|||이모지 쌍 ```(이모지A, 이모지B)``` 별 코사인 유사도 평균값 계산|