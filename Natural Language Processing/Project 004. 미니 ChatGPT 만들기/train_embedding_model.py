from embedding_helper import get_token_ids, encode_one_hot
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# fastest model from https://www.sbert.net/docs/pretrained_models.html
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
embeddings = {}


def embed_text(text):
    if text not in embeddings:
        embeddings[text] = sbert_model.encode(text)
    return embeddings[text]
        

# train_data.csv 파일의 데이터를 S-BERT 임베딩으로 변환해서 반환
# 입력 : (2번째 단어의 S-BERT embedding) * A
# 출력 : (1번째, 3번째 단어의 S-BERT embedding vector 의 평균) * B
def get_train_data_as_embeddings(verbose=False):
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
    
    input_data = []
    output_data = []
    train_data_iter_cnt = 0

    # embedding을 "표준정규분포에 가까운" input, output data 로 변환하기 위해 곱하는 값 A, B
    A = 2.5
    B = 3.5

    for _, row in train_data.iterrows():
        if train_data_iter_cnt % 1500 == 0:
            print(f'count: {train_data_iter_cnt}')
        
        tokens = row['data'].split(' ')
        second_token_input = embed_text(tokens[1])
        first_token_output = embed_text(tokens[0])
        third_token_output = embed_text(tokens[2])
        
        input_data.append(second_token_input * A)
        output_data.append((first_token_output + third_token_output) / 2.0 * B)
        
        train_data_iter_cnt += 1

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    print(f'input data (shape: {np.shape(input_data)}) :\n{input_data}')
    print(f'output data (shape: {np.shape(output_data)}) :\n{output_data}')

    return input_data, output_data


if __name__ == '__main__':
    input_data, output_data = get_train_data_as_embeddings(verbose=True)

