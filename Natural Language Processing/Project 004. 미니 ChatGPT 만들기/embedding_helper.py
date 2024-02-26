import pandas as pd
import numpy as np


# 단어를 사전순으로 정렬하여 첫 단어부터 0, 1, ... 로 매긴 dict 반환
# 예: {'a': 0, 'about': 1, 'and': 2, ...}
def get_token_ids():
    train_data = pd.read_csv('train_data.csv', index_col=0)
    token_list = []

    for idx, row in train_data.iterrows():
        tokens = row['data'].split(' ')
        
        if idx == len(train_data) - 1:
            for token in tokens:
                token_list.append(token)
                
        else:
            token_list.append(tokens[0])

    token_list = list(set(token_list))
    token_list.sort()

    token_ids = {}
    for i, token in enumerate(token_list):
        token_ids[token] = i

    return token_ids
        

# token 을 one-hot embedding한 배열 반환
def encode_one_hot(token_ids, token):
    result = [0 for i in range(len(token_ids))]
    result[token_ids[token]] = 1
    return result

