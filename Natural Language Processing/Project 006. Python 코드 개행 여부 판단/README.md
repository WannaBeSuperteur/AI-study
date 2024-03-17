# NLP Project 006. Python 코드 개행 여부 판단
* 목표: **Python 코드에서, 그 직전 및 직후의 line들의 내용을 근거로, 특정한 2개 line 사이에 개행 (=비어 있는 한 줄) 이 필요한지를 판단하는 AI 모델을 만든다.**
* Dataset: [Python Code Data](https://www.kaggle.com/datasets/veeralakrishna/python-code-data)
  * 파이썬 코드 전처리 및 tokenize 기준
    * 파이썬 코드에서 각 변수명은 ```var1```, ```var2```, ... 로 대체한다.
    * 파이썬 코드에서 각 함수명은 ```func1```, ```func2```, ... 로 대체한다.
    * 파이썬 코드에서 숫자 값은 ```(n)``` 으로 대체한다.
    * ```''```, ```""``` 안에 들어가는 텍스트는 각각 ```mytxttkn1```, ```mxtxttkn2```, ... 로 대체한다. (서로 같은 텍스트는 동일한 이름으로 처리)
    * 괄호, 연산기호, ```if```, ```and```, 개행, 각 변수명, 각 함수명 등을 모두 token으로 간주한다.
  * AI 실행 과정
    * ```python_code.txt``` 파일의 코드의 **N번째 줄과 N+1번째 줄** 사이에 개행이 필요한지 판단하기 위해 다음과 같이 한다.
      * N-1, N-2 번째 줄을 각각 tokenize 한다.
      * N+1, N+2 번째 줄을 각각 tokenize 한다.
      * 이상의 4개의 줄을 tokenize 한 결과를 입력으로, 개행 여부를 출력으로 하는 모델 ```main_model``` 에 해당 4개의 line을 입력해서, 모델의 출력값으로 개행 여부를 결정한다. 
* 수행 기간: 2024.03.17 ~ 2024.03.19 (3일)
* 참고: **일부 코드는 [Python 코드 자동 수정 프로젝트](https://github.com/WannaBeSuperteur/AI-study/tree/GAI-P2-2/Natural%20Language%20Processing/Project%20003.%20Python%20%EC%BD%94%EB%93%9C%20%EC%9E%90%EB%8F%99%20%EC%88%98%EC%A0%95) 의 것을 그대로 복사해서 가져왔습니다.**

## 파일 및 코드 설명
* ```Python_code_data.txt``` (데이터셋을 다운받아야 함) : 학습 데이터
* ```read_code.py``` : 데이터를 읽어서 각 code snippet을 수집하는 함수
  * 출력 파일 : ```code_snippets.csv``` (code snippet을 수집한 결과)
* ```tokenize.py``` : 수집된 code snippet 들을 tokenize 하여 모델 입력 및 출력 데이터로 저장하는 함수
  * 필요 파일 : ```code_snippets.csv``` 
  * 출력 파일 : ```train_data.csv``` (모델 학습용 입력 및 출력 데이터, token의 값이 그대로 쓰여 있음)
  * 출력 파일 : ```train_data_token_id.csv``` (모델 학습용 입력 및 출력 데이터, token ID가 있음) 
* ```train.py``` : 개행 여부 판단 모델 ```main_model``` 을 학습하는 함수
  * 필요 파일 : ```train_data_token_id.csv```
  * 출력 파일 : ```train_data_token_id_for_test.csv``` (테스트용 데이터)
  * 출력 모델 : ```main_model```
* ```test.py``` : ```main_model``` 을 테스트하고 성능지표 결과를 출력하는 함수
  * 필요 모델 : ```main_model```
  * 필요 파일 : ```train_data_token_id_for_test.csv```

## 실행 순서
```
python read_code.py
python tokenize.py
python train.py
python test.py
```

## 성능지표 결과
```test.py``` 파일을 실행시켜서 다음을 측정 **(개행해야 하는 경우를 positive 로 간주)**
* Accuracy (정확도)
* Recall ```= TP / (TP + FN)```
* Precision ```= TP / (TP + FP)```
* F1 Score ```= 2 * Recall * Precision / (Recall + Precision)```

개념 설명
* TP = True Positive (예측이 positive, 실제 값도 positive인 것)
* FN = False Negative (예측은 negative, 실제 값은 positive인 것)
* FP = False Positive (예측은 positive, 실제 값은 negative인 것)

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|NLP-P6-master|||240317|240319|마스터 브랜치|
|NLP-P6-1|```done```|```feat```|240317|240317|code snippet을 수집하여 저장|
|NLP-P6-2|```done```|```feat```|240317|240317|수집된 code snippet을 tokenize 하여 모델 학습용 데이터로 저장|
|NLP-P6-3|```ing```|```feat```|240317||개행 여부 판단 모델인 ```main_model``` 학습|
|NLP-P6-4||```feat```|||개행 여부 판단 모델인 ```main_model``` 테스트|