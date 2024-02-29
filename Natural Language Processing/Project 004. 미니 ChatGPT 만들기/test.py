from embedding_helper import get_token_ids
from train_main_model import test_main_model
from train_embedding_model import test_embedding_model
import math
import tensorflow as tf
import numpy as np

EMBEDDING_DIM = 24


def tokenize_for_test(text):

    # 문장 부호 및 축약형 처리
    text = text.replace(',', ' ,').replace('.', ' .')
    text = text.replace('!', ' !').replace('?', ' ?')
    text = text.replace("'m", " 'm").replace("'re", " 're").replace("'s", " 's")
    text = text.replace("'t", " 't").replace("'ve", " 've").replace("'d", " 'd")
    text = text.replace('"', '')

    # 소문자 처리
    text = text.lower()

    # 발화자 구분 처리
    text = text.replace('<next>', '<Person-Change>')
    
    return text


# 사용자 입력 (must be 16 tokens)
def get_user_input():
    while True:
        input_text = input('\ninput text with 16 tokens.\n발화자 구분: <NEXT> (예: How are you? <NEXT> I am fine.)\n')
        input_text_tokenized = tokenize_for_test(input_text + ' <NEXT>')
        
        token_count = len(input_text_tokenized.split(' '))
        if token_count != 16:
            print(f'\nThe number of token must be 16. The number of token in your input is {token_count}.')
        else:
            return input_text_tokenized


# 각 token 별 본 프로젝트의 임베딩 모델의 임베딩 계산
def compute_embedding(token_ids):
    embedding_candidates = {}

    embedding_model = tf.keras.models.load_model('embedding_model')
    embedding_encoder = embedding_model.encoder
    count = 0
    
    for token, _ in token_ids.items():
        if count % 250 == 0:
            print(count)
            
        embedding_candidates[token] = test_embedding_model(
            token,
            is_return=True,
            embedding_encoder=embedding_encoder,
            verbose=False
        )

        count += 1

    return embedding_candidates


# Euclidean Distance
def compute_euclidean_distance(a, b, weight):
    n = len(a)
    result = 0

    for i in range(n):
        result += (weight[i] * (a[i] - b[0][i])) ** 2

    return math.sqrt(result)


# 각 token 과의 거리 계산
def compute_distance(next_token_embedding, candidates, weight):
    distances = []
    count = 0

    for token, emb in candidates.items():
        if count % 250 == 0:
            print(count)
            
        dist = compute_euclidean_distance(next_token_embedding, emb, weight)
        distances.append([token, dist])

        count += 1

    distances.sort(key=lambda x:x[1])
    return distances


if __name__ == '__main__':
    token_ids = get_token_ids()
    embedding_candidates = compute_embedding(token_ids)
    
    weight = np.random.uniform(0.5, 1.5, EMBEDDING_DIM)

    # 사용자 입력
    tokenized_input = get_user_input()

    outputs = []

    for i in range(12):
        print(f'tokenized input : {tokenized_input}')
        
        # 다음 토큰 예측 임베딩 (메인 모델 학습 시 output에 x2.5를 했으므로, 먼저 /2.5를 해서 복구해야 함)
        next_token_embedding = test_main_model(tokenized_input, is_return=True)[0] / 2.5

        # 가장 가까운 토큰 계산 (with weight)
        distances = compute_distance(next_token_embedding, embedding_candidates, weight)
        print(f'weight: {weight}')

        for j in range(10):
            print(distances[j])

        outputs.append(distances[0][0])
        tokenized_input = ' '.join(tokenized_input.split(' ')[1:]) + ' ' + distances[0][0]

    print('outputs:', ' '.join(outputs))
