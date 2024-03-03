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
def cos_sim(x, y):
    return dot(x, y) / (norm(x) * norm(y))


# embedding array A를 이용한 임베딩 검증 (임시)
def validate_embedding(A, token_ids, token_pairs):
    for token_pair in token_pairs:
        first_token_embedding = A[token_ids[token_pair[0]]]
        second_token_embedding = A[token_ids[token_pair[1]]]
        
        cos_similarity = cos_sim(first_token_embedding, second_token_embedding)
        print(f'embedding between {token_pair[0]}, {token_pair[1]} : {cos_similarity}')


# embedding array A를 이용한 가장 가까운 토큰 순위 표시
def print_most_similar_tokens(A, token_ids, token_arr, token):
    token_embedding = A[token_ids[token]]

    result = []
    for i in range(len(A)):
        token_to_check_sim = token_arr[i]
        emb_of_token = A[i]
        cos_similarity = cos_sim(token_embedding, emb_of_token)
        
        result.append([token_to_check_sim, cos_similarity])

    result.sort(key=lambda x: x[1], reverse=True)

    print(f'\n10 nearest token of "{token}" based on embedding:')
    for i in range(11):
        print(f'rank {i} : {result[i]}')
