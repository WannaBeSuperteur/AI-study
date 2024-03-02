from embedding_helper import get_token_ids, get_token_arr, encode_one_hot
import pandas as pd
import numpy as np
import os

from tokenize_data import get_maps, tokenize_line

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, LeakyReLU, Dropout


INPUT_TOKEN_CNT = 36 # 학습 데이터 row 당 입력 토큰 개수
TKN_EMBEDDING_DIM = 128 # token embedding dimension
VOCAB_SIZE = None


# mini chatgpt model (=NLP model)
# ref: https://www.kaggle.com/code/carlosaguayo/predicting-the-next-word-using-lstm/notebook
class MiniChatGPTModel(tf.keras.Model):
    
    def __init__(self, dropout_rate=0.45):
        super().__init__()
        global VOCAB_SIZE, INPUT_TOKEN_CNT

        L2 = tf.keras.regularizers.l2(0.001)
        self.dropout = Dropout(rate=dropout_rate)

        # token embedding
        self.tkn_embedding = Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=TKN_EMBEDDING_DIM,
            input_length=INPUT_TOKEN_CNT
        )
        
        self.BIDRC_LSTM_0 = Bidirectional(LSTM(128))
        self.dense = Dense(192, activation=LeakyReLU(alpha=0.1))
        self.final = Dense(VOCAB_SIZE, activation='softmax')

    def call(self, inputs, training):
        positions = tf.range(start=0, limit=INPUT_TOKEN_CNT, delta=1)
        
        embed_tkn = self.tkn_embedding(inputs)

        intermediate_0 = self.dropout(embed_tkn)
        intermediate_0 = self.BIDRC_LSTM_0(intermediate_0)

        intermediate_1 = self.dropout(intermediate_0)
        intermediate_1 = self.dense(intermediate_1)
        
        outputs = self.final(intermediate_1)
        return outputs


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6) # RMSProp 적용 시 train loss 발산 (확인 필요)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=500)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=40, factor=0.5)
        
    model = MiniChatGPTModel()
    return model, optimizer, early_stopping, lr_reduced


# input data, output data 로부터 train, valid data 추출
def define_data(input_data_all, output_data_all, valid_ratio=0.1):
    
    valid_cnt = int(valid_ratio * len(input_data_all))

    # define train and valid, input and output data
    train_input_data = input_data_all[:-valid_cnt]
    train_output_data = output_data_all[:-valid_cnt]

    valid_input_data = input_data_all[-valid_cnt:]
    valid_output_data = output_data_all[-valid_cnt:]

    return train_input_data, train_output_data, valid_input_data, valid_output_data



# NLP 모델 학습
def train_model(input_data_all, output_data_all):
    (train_input, train_output, valid_input, valid_output) = define_data(input_data_all, output_data_all)

    print(f'train input : {np.shape(train_input)}\n{train_input}\n')
    print(f'valid input : {np.shape(valid_input)}\n{valid_input}\n')
    print(f'train output : {np.shape(train_output)}\n{train_output}\n')
    print(f'valid output : {np.shape(valid_output)}\n{valid_output}\n')
    
    model, optimizer, early_stopping, lr_reduced = define_model()
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=180,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('mini_chatgpt_model')

    return model


# train_data.csv 파일의 데이터 전체를 token id로 만들어서 반환
# for example,
"""
input : (44380, 24)
[[1174    1   77 ...   77 2764 2735]
 [  76   76   76 ...   77 2764 2735]
 [   1   77 2764 ... 2764 2735 2514]
 ...
 [  76   76   76 ... 1323   11 1325]
 [1734 2269   25 ...   11 1325   25]
 [  76   76   76 ...   76   76   76]]

output : (44380, 1)
[[2514]
 [2514]
 [1607]
 ...
 [  25]
 [  77]
 [  77]]
"""

def get_train_data_token_ids(token_ids, verbose=False, limit=None):
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

    train_data_iter_cnt = 0

    input_data_all = [] # 전체 학습 입력 데이터
    output_data_all = [] # 전체 학습 출력 데이터

    for _, row in train_data.iterrows():
        if train_data_iter_cnt % 5000 == 0:
            print(f'count: {train_data_iter_cnt}')

        if limit is not None and train_data_iter_cnt >= limit:
            break
        
        input_tokens = row['data'].split(' ')[:input_size]
        output_token = row['data'].split(' ')[-1]
        input_text_token_ids = []
        output_text_token_ids = []

        for input_token in input_tokens:
            input_text_token_ids.append(token_ids[input_token])
        output_text_token_ids.append(encode_one_hot(token_ids=token_ids, token=output_token))

        train_data_iter_cnt += 1

        # flatten 해서 전체 데이터에 추가
        input_data_all.append(np.array(input_text_token_ids).flatten())
        output_data_all.append(np.array(output_text_token_ids).flatten())

    return np.array(input_data_all), np.array(output_data_all)


# 모델 학습 프로세스 전체 진행
def run_all_process(limit=None):
    global VOCAB_SIZE
    
    token_ids = get_token_ids()

    # vocab size 초기화
    VOCAB_SIZE = len(token_ids)
    print(f'VOCAB SIZE = {VOCAB_SIZE}')
    
    # 학습 데이터
    input_data_all, output_data_all = get_train_data_token_ids(token_ids, limit=limit)

    # 모델 학습 및 저장
    main_model = train_model(input_data_all, output_data_all)
    return main_model


# NLP 모델 테스트
def test_model(text, model, additional_tokenize=True, is_return=False, verbose=False, token_arr=None, token_ids=None):
    if token_ids is None:
        token_ids = get_token_ids()

    if additional_tokenize:
        ing_map, ly_map = get_maps()
        tokenized_line = tokenize_line(text, ing_map, ly_map)
        tokens = (tokenized_line.split(' ') + ['<person-change>'])
    else:
        tokens = text.split(' ')

    if verbose:
        print(f'\ntokens: {tokens}')

    tokens_id = [token_ids[t] for t in tokens]

    if verbose:
        print(f'ID of tokens: {tokens_id}')

    if len(tokens_id) < INPUT_TOKEN_CNT:
        rest = INPUT_TOKEN_CNT - len(tokens_id)
        tokens_id = ([token_ids['<null>']] * rest) + tokens_id

    if verbose:
        print(f'ID of tokens: {tokens_id}')

    output = model([tokens_id])

    if verbose:
        print(f'mini chatgpt model output: {output[0]}')

    output_rank = []

    if token_arr is None:
        token_arr = get_token_arr()

    if verbose:
        print(f'first 10 of token arr: {token_arr[:10]}')

    for i in range(len(token_ids)):
        token = token_arr[i]
        prob = float(output[0][i])
        output_rank.append([token, prob])
            
    output_rank.sort(key=lambda x:x[1], reverse=True)

    if verbose:
        for i in range(20):
            print(f'rank {i} : {output_rank[i]}')

    if is_return:
        return output_rank
    

if __name__ == '__main__':

    # 전체 실행
    if 'mini_chatgpt_model' not in os.listdir():
        run_all_process()

    # 메인 모델 테스트 (each example text has 16 tokens)
    example_texts = [
        'what was the most number of people you have ever met during a working day ?',
        'i know him very well . <person-change> is him your friend ? if so , it',
        'how can i do for you ? <person-change> can you buy me a book ?'
    ]

    mini_chatgpt_model = tf.keras.models.load_model('mini_chatgpt_model')
    
    for example_text in example_texts:
        try:
            test_model(example_text, model=mini_chatgpt_model, verbose=True)
        except Exception as e:
            print(f'error: {e}')
