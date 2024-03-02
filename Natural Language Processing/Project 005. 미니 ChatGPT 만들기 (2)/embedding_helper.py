import pandas as pd
import numpy as np
import tensorflow as tf

from sentence_transformers import SentenceTransformer

# fastest model from https://www.sbert.net/docs/pretrained_models.html
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

SBERT_EMBED_SIZE_TO_USE = 128


# 단어를 사전순으로 정렬하여 첫 단어부터 0, 1, ... 로 매긴 dict 반환
# 예: {'a': 0, 'about': 1, 'and': 2, ...}
def get_token_ids():
    train_data = pd.read_csv('train_data.csv', index_col=0)
    token_list = []
    row_idx = 0

    for _, row in train_data.iterrows():
        tokens = row['data'].split(' ')

        # 마지막 행 또는 다음 행이 <null> 로 시작하는 행이면
        if row_idx >= len(train_data) - 1 or train_data.iloc[row_idx + 1]['data'].startswith('<null>'):
            for token in tokens:
                token_list.append(token)
                
        else:
            token_list.append(tokens[0])

        row_idx += 1

    token_list = list(set(token_list))
    token_list.sort()

    token_ids = {}
    for i, token in enumerate(token_list):
        token_ids[token] = i

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


# 임베딩 모델의 encoder (임베딩을 하는 부분) 로딩
def load_embedding_encoder():
    embedding_model = tf.keras.models.load_model('embedding_model')
    embedding_encoder = embedding_model.encoder
    return embedding_encoder


# 각 token에 대한, 본 프로젝트에서 개발한 임베딩 모델에 의한 임베딩 배열 (크기 24)
def get_embedding_of_token(token, embedding_encoder):
    sbert_embedding = sbert_model.encode(token)[:SBERT_EMBED_SIZE_TO_USE]
    model_embedding = embedding_encoder(np.array([sbert_embedding]))
    return model_embedding
