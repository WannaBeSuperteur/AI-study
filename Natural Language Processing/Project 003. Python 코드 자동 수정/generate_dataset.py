import os
import tensorflow as tf
import pandas as pd
from embedding_cbow import run_embedding
import numpy as np


vocab_n = 94 # embedding_cbow.py 의 vocab 구조 수정 시 업데이트
window_size = 4
embedding_size = 16


# 실행 시, "embedding_dataset.csv" 파일 필요
# 현재 시점 기준으로 window size = N = 4


# 각 vocab idx 별 임베딩 계산 후 반환
def get_embeddings_for_each_idx(emb_model):
    vocab_idx_embeddings = []

    for i in range(vocab_n):
        embed_result = run_embedding(i, emb_model.encoder)
        vocab_idx_embeddings.append(list(embed_result))

    return np.array(vocab_idx_embeddings)


# embedding DataFrame 변환
# word index from vocab -> embedding result 
def convert_embedding_df(embedding_df, embeddings):
    for i in range(window_size):
        for j in range(embedding_size):
            print(f'processing window {i}, embedding vector index {j} (before) ...')
            
            embedding_df[f'ib_{i}_{j}'] = embedding_df.apply(
                lambda x: embeddings[int(x[f'ib_{i}'])][j],
                axis=1
            )

        for j in range(embedding_size):
            print(f'processing window {i}, embedding vector index {j} (after) ...')
            
            embedding_df[f'ia_{i}_{j}'] = embedding_df.apply(
                lambda x: embeddings[int(x[f'ia_{i}'])][j],
                axis=1
            )
            
        embedding_df.drop(columns=[f'ib_{i}'], inplace=True)
        embedding_df.drop(columns=[f'ia_{i}'], inplace=True)

    for i in range(embedding_size):
        print(f'processing embedding vector index {i} (output) ...')
            
        embedding_df[f'output_{i}'] = embedding_df.apply(
            lambda x: embeddings[int(x['out'])][i],
            axis=1
        )

    for i in range(vocab_n):
        print(f'processing vocab index {i} for output ...')
        
        embedding_df[f'out_{i}'] = embedding_df.apply(lambda x: 1.0 if x['out'] == i else 0.0, axis=1)

    embedding_df.drop(columns=['out'], inplace=True)


# 변환된 embedding DataFrame에서 데이터셋 생성
def create_dataset(embedding_df):
    print(embedding_df)
    print(f'columns: {list(embedding_df.columns)}')
    
    embedding_train_df_0 = pd.DataFrame(embedding_df)
    
    embedding_train_df_1 = pd.DataFrame()
    n = len(embedding_train_df_0)

    for i in range(int(n * 0.5)):
        if i % 100 == 0:
            print(f'{i} / {n * 0.5}')
            
        new_row = {}
        for j in range(window_size):
            for k in range(embedding_size):
                new_row[f'ib_{j}_{k}'] = [float(embedding_train_df_0.iloc[i][f'ib_{j}_{k}'])]

        for k in range(embedding_size):
            new_row[f'ia_0_{k}'] = [float(embedding_train_df_0.iloc[i][f'output_{k}'])]

        for j in range(window_size - 1):
            for k in range(embedding_size):
                new_row[f'ia_{j + 1}_{k}'] = [float(embedding_train_df_0.iloc[i][f'ia_{j}_{k}'])]

        for j in range(vocab_n):
            new_row[f'out_{j}'] = [0.0]

        new_row = pd.DataFrame(new_row)
        embedding_train_df_1 = pd.concat([embedding_train_df_1, new_row])

    # remove unused output_x columns
    for i in range(embedding_size):
        embedding_train_df_0.drop(columns=[f'output_{i}'], inplace=True)

    final_dataset = pd.concat([embedding_train_df_0, embedding_train_df_1])
    return final_dataset


# main model의 학습 데이터 파일 찾기
def get_or_create_main_model_train_data():
    if 'main_model_train_df.csv' not in os.listdir():
        embedding_df = pd.read_csv('embedding_dataset.csv', index_col=0)
        emb_model = tf.keras.models.load_model('embedding_model')

        embeddings = get_embeddings_for_each_idx(emb_model)
        convert_embedding_df(embedding_df, embeddings)

        embedding_final_df = create_dataset(embedding_df)
        embedding_final_df.to_csv('main_model_train_df.csv')

    return pd.read_csv('main_model_train_df.csv', index_col=0)


if __name__ == '__main__':
    print(get_or_create_main_model_train_data())

    
