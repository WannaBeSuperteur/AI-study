from embedding_helper import get_token_ids, encode_one_hot
import pandas as pd
import numpy as np


# train_data.csv 파일의 데이터 전체를 one-hot 배열로 만들어서 반환
def get_train_data_one_hot(verbose=False):
    token_ids = get_token_ids()
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
            train_data_one_hot_row += encode_one_hot(token_ids, tokens[i])
        train_data_one_hot.append(train_data_one_hot_row)

        train_data_iter_cnt += 1

    return np.array(train_data_one_hot)


if __name__ == '__main__':
    train_data_one_hot = get_train_data_one_hot(verbose=True)
    print(train_data_one_hot)
