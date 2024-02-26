from embedding_helper import get_token_ids, load_embedding_encoder, get_embedding_of_token
import pandas as pd
import numpy as np

embedding_dic = {}


# 각 token을 본 프로젝트에서 개발한 임베딩 모델에 의한 embedding으로 매핑
# example of 'token_ids': {'a': 0, 'about': 1, 'and': 2, ...}
def fill_embedding_dic(token_ids):
    embedding_encoder = load_embedding_encoder()
    
    for token, idx in token_ids.items():
        if token not in embedding_dic:
            embedding_dic[token] = get_embedding_of_token(token, embedding_encoder)

            if token in ['am', 'are', 'is', 'i', 'you', 'he', 'she']:
                print(f'token: {token}\nembedding:\n{embedding_dic[token]}')


# train_data.csv 파일의 데이터 전체를,
# 본 프로젝트에서 개발한 임베딩 모델에 의한 임베딩 배열로 만들어서 반환
def get_train_data_embedding(token_ids, verbose=False):
    vocab_size = len(token_ids)
    
    train_data = pd.read_csv('train_data.csv', index_col=0)
    input_size = len(train_data.iloc[0]['data'].split(' ')) - 1
    output_size = 1
    all_size = input_size + output_size

    if verbose:
        print(f'vocab  size: {vocab_size}')
        print(f'input  size: {input_size}')
        print(f'output size: {output_size}')
        print(f'all    size: {all_size} ( = input + output)')
        print(f'data   size: {len(train_data)}')
    
    train_data_one_hot = []
    train_data_iter_cnt = 0

    for _, row in train_data.iterrows():
        if train_data_iter_cnt % 2500 == 0:
            print(f'count: {train_data_iter_cnt}')
        
        tokens = row['data'].split(' ')

        train_data_one_hot_row = []
        for i in range(all_size):
            train_data_one_hot_row += list(embedding_dic(tokens[i]))
        train_data_one_hot.append(train_data_one_hot_row)

        train_data_iter_cnt += 1

    return np.array(train_data_one_hot)


if __name__ == '__main__':
    token_ids = get_token_ids()

    # 각 token을 본 프로젝트에서 개발한 임베딩 모델에 의한 embedding으로 매핑 
    fill_embedding_dic(token_ids)

    # 학습 데이터
    train_data_embedding = get_train_data_embedding(token_ids, verbose=True)
    
    print(train_data_one_hot)
