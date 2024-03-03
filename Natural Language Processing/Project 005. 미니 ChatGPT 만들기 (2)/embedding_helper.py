import pandas as pd
import numpy as np
import tensorflow as tf
import time
from numpy import dot
from numpy.linalg import norm


# 단어를 사전순으로 정렬하여 첫 단어부터 0, 1, ... 로 매긴 dict 반환
# 예: {'a': 0, 'about': 1, 'and': 2, ...}
def get_token_ids():
    start_time = time.time()
    train_data = pd.read_csv('train_data.csv', index_col=0)
    token_list = []

    for _, row in train_data.iterrows():
        tokens = row['data'].split(' ')
        for token in tokens:
            token_list.append(token)

    token_list = list(set(token_list))
    token_list.sort()

    token_ids = {}
    for i, token in enumerate(token_list):
        token_ids[token] = i

    print(f'{time.time() - start_time} seconds for getting token ids')
    return token_ids


# 단어의 순서를 나타낸 배열 반환 (get_token_ids() 결과의 inverse)
# 예: ['a', 'about', 'and', ...]
def get_token_arr():
    token_ids = get_token_ids()
    vocab_size = len(token_ids)
    result = ['' for _ in range(vocab_size)]

    for token, idx in token_ids.items():
        result[idx] = token

    return result
    

# token 을 one-hot embedding한 배열 반환
def encode_one_hot(token_ids, token):
    result = [0 for i in range(len(token_ids))]
    result[token_ids[token]] = 1
    return result


# cosine similarity
def cos_sim(x, y, weight=None):
    n = len(x)

    if weight is not None:
        x_weighted = [x[i] * weight[i] for i in range(n)]
        y_weighted = [y[i] * weight[i] for i in range(n)]
        return dot(x_weighted, y_weighted) / (norm(x_weighted) * norm(y_weighted))
    else:
        return dot(x, y) / (norm(x) * norm(y))



