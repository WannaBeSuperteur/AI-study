# NLP Project 002. 이모지 데이터 유사도 검증
* Dataset: [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
  * 문장과 이모지 표정이 ```;``` 로 구분된 텍스트의 집합
* 수행 기간: 2024.02.07 ~ 02.08 (2일)

## License and Acknowledgements about the dataset
* LICENSE : **CC BY-SA 4.0** [라이선스 설명](https://creativecommons.org/licenses/by-sa/4.0/)
* Thanks to : **Elvis (https://lnkd.in/eXJ8QVB) & Hugging Face**
* 데이터셋 제작에 사용된 기술 : https://www.aclweb.org/anthology/D18-1404/ 논문 참고
  * Citation : ```Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. 2018. CARER: Contextualized Affect Representations for Emotion Recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3687–3697, Brussels, Belgium. Association for Computational Linguistics.``` 

## 파일 및 코드 설명
* ```train.txt``` (데이터셋을 다운받아야 함) : 학습 데이터 (```val.txt```, ```test.txt``` 는 사용하지 않음)
* ```read_data.py``` : ```train.txt``` 데이터를 읽어서 pandas DataFrame 으로 변환
* ```embed_text.py``` : 텍스트 임베딩 (Sentence BERT를 이용) 및 관련 util 함수
* ```main.py``` : 데이터를 읽어서, 문장 2개 추출, 임베딩, 비교분석 시행 반복 후, 각 이모지 쌍에 대한 코사인 유사도 평균값 계산
  * 문장, 이모지, 임베딩 쌍 파일 : ```train_embed.csv```
  * 문장 쌍 비교 실험 결과 파일 : ```emoji_and_embedding.csv```
  * 이모지 쌍별 코사인 유사도 평균값 계산 결과 : ```avg_cos_sim.csv```

## 코드 실행 순서
* ```main.py``` 단독 실행 (```python main.py``` 또는 ```python3 main.py```)

## 작업 설명
* 각 문장을 Sentence-BERT 를 이용하여 임베딩한다.
* 임베딩된 각 문장 중 랜덤으로 2개를 추출하는 시행을 N회 반복한다.
  * 이때, 각 문장 간 Sentence-BERT 임베딩의 코사인 유사도와, 해당 문장에 대응되는 이모지가 일치하는지의 여부를 확인한다.
* 두 문장의 이모지가 각각 ```이모지A```, ```이모지B``` (단, ```이모지A```와 ```이모지B```는 서로 다름) 인 모든 경우에 대해, 해당 두 문장의 Sentence-BERT 임베딩 결과의 코사인 유사도에 대한 평균값을 구한다.
  * 이를 통해 각 이모지가 Sentence-BERT 임베딩 기준 얼마나 의미적으로 서로 유사한지 판단한다.

## 실험 결과
* 코사인 유사도의 평균값 기준, **N (추출 반복횟수) = 10,000**

|이모지A \ 이모지B|anger|fear|joy|love|sadness|surprise|
|---|---|---|---|---|---|---|
|anger|**0.261**|0.250|0.203|0.218|0.257|0.227|
|fear|0.250|0.254|0.204|0.191|**0.261**|0.230|
|joy|0.203|0.204|0.217|0.216|0.208|**0.222**|
|love|0.218|0.191|0.216|0.212|**0.219**|0.200|
|sadness|0.257|0.261|0.208|0.219|**0.263**|0.231|
|surprise|0.227|0.230|0.222|0.200|0.231|**0.343**|

* ```anger```, ```sadness```, ```surprise``` 이모지에 해당하는 문장들은 동일한 이모지의 문장들과의 코사인 유사도 평균값이 나머지 모든 이모지보다 높다.
* ```fear```, ```love``` 이모지의 문장들은 모두 ```sadness``` 이모지 문장들과의 코사인 유사도 평균값이 나머지 모든 이모지보다 **(동일 이모지보다도)** 높다. (역시 사랑은 결국 슬픈 것이다.)
* ```joy``` 이모지의 문장들은 ```surprise``` 이모지 문장들과의 코사인 유사도 평균값이 나머지 모든 이모지보다 **(동일 이모지보다도)** 높다.
* 추출한 표본이 100건 미만인 이모지 쌍은 다음과 같다.
  * ```(fear, surprise)``` (82건), ```(love, love)``` (59건), ```(love, surprise)``` (57건), ```(surprise, surprise)``` (12건)

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|NLP-P2-master|||240207|240208|마스터 브랜치|
|NLP-P2-1|```done```|```feat```|240207|240207|데이터 읽기|
|NLP-P2-2|```done```|```feat```|240207|240207|텍스트 임베딩 및 임베딩 관련 함수 작성|
|NLP-P2-3|```done```|```feat```|240207|240208|임베딩된 문장 2개 추출 시행 반복 및 결과 저장|
|NLP-P2-4|```done```|```feat```|240208|240208|이모지 쌍 ```(이모지A, 이모지B)``` 별 코사인 유사도 평균값 계산|
