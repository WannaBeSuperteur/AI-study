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
