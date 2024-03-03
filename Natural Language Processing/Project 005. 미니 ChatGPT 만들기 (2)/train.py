from embedding_helper import get_token_ids, get_token_arr, encode_one_hot
from add_bert_embedding_dict import find_embedding_from_table, find_nearest_bert_embedding_rank_for_embvec
import pandas as pd
import numpy as np
import os

from tokenize_data import get_maps, tokenize_line

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, LeakyReLU, Dropout, Flatten, concatenate
from tensorflow.keras.utils import get_custom_objects


INPUT_TOKEN_CNT_EACH = 30 # 학습 데이터 row 당 각 발화자의 turn에 대한 입력 토큰 개수
TKN_EMBEDDING_DIM = 128 # token embedding dimension
VOCAB_SIZE = None


# 사용자 정의 activation function = 1.5 * tanh(x) (-1 < tanh(x) < 1 인데, output 되는 값의 절댓값이 1보다 큰 경우 존재)
class Tanh_mul(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Tanh_mul, self).__init__(activation, **kwargs)
        self.__name__ = 'Tanh_mul'
        
def tanh_mul(x):
    return 1.5 * K.tanh(x)

get_custom_objects().update({'tanh_mul': Tanh_mul(tanh_mul)})


# mini chatgpt model (=NLP model)
# ref: https://www.kaggle.com/code/carlosaguayo/predicting-the-next-word-using-lstm/notebook
class MiniChatGPTModel(tf.keras.Model):
    
    def __init__(self, dropout_rate=0.45):
        super().__init__()
        global VOCAB_SIZE, INPUT_TOKEN_CNT_EACH

        L2 = tf.keras.regularizers.l2(0.001)
        self.dropout = Dropout(rate=dropout_rate)

        # token embedding
        self.tkn_embedding = Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=TKN_EMBEDDING_DIM,
            input_length=INPUT_TOKEN_CNT_EACH
        )

        # LSTM
        self.BIDRC_LSTM_LAST = Bidirectional(LSTM(64))
        self.BIDRC_LSTM_CURRENT = Bidirectional(LSTM(64))

        # last turn, current turn Dense
        self.last_turn_dense = Dense(48, activation=LeakyReLU(alpha=0.1))
        self.current_turn_dense = Dense(48, activation=LeakyReLU(alpha=0.1))

        # dense layers
        self.dense = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.final = Dense(TKN_EMBEDDING_DIM, activation='tanh_mul')


    def call(self, inputs, training):
        inputs_last_turn, inputs_current_turn = tf.split(inputs, [INPUT_TOKEN_CNT_EACH, INPUT_TOKEN_CNT_EACH], axis=1)

        # for last turn
        embed_tkn_last_turn = self.tkn_embedding(inputs_last_turn)
        lstm_last_turn = self.BIDRC_LSTM_LAST(embed_tkn_last_turn)
        dense_last_turn = self.last_turn_dense(lstm_last_turn)

        # for current turn
        embed_tkn_current_turn = self.tkn_embedding(inputs_current_turn)
        lstm_current_turn = self.BIDRC_LSTM_CURRENT(embed_tkn_current_turn)
        dense_current_turn = self.current_turn_dense(lstm_current_turn)

        # concatenation, ...
        AB_concat = tf.keras.layers.Concatenate()([dense_last_turn, dense_current_turn])
        AB_dense = self.dense(AB_concat)

        # final output
        outputs = self.final(AB_dense)
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
    print(f'all input : {np.shape(input_data_all)}\n{input_data_all}\n')
    print(f'all output : {np.shape(output_data_all)}\n{output_data_all}\n')
    
    (train_input, train_output, valid_input, valid_output) = define_data(input_data_all, output_data_all)

    print(f'train input : {np.shape(train_input)}\n{train_input}\n')
    print(f'valid input : {np.shape(valid_input)}\n{valid_input}\n')
    print(f'train output : {np.shape(train_output)}\n{train_output}\n')
    print(f'valid output : {np.shape(valid_output)}\n{valid_output}\n')
    
    model, optimizer, early_stopping, lr_reduced = define_model()
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=3,
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

output : (44380, embedding_dim) = (44380, 128), first 128 elements of BERT embedding of output token
(단, 여기에 표시된 각 token ID에 대한 BERT 임베딩은 임의로 만든 것으로, 아래와 다를 수 있음)

[[2514]      [[0.53, 0.45, -0.17, ...,  0.28,  0.19, 0.6 ]
 [2514]       [0.53, 0.45, -0.17, ...,  0.28,  0.19, 0.6 ]
 [1607]       [1.12, 0.7,   0.13, ..., -0.45, -0.79, 0.02]
 ...      =>  ...
 [  25]       [0.97, 0.08,  0.02, ..., -1.01,  0.9 , 0.44]
 [  77]       [0.66, 0.24,  0.19, ...,  0.38, -0.99, 0.75]
 [  77]]      [0.66, 0.24,  0.19, ...,  0.38, -0.99, 0.75]]
"""

def get_train_data_token_embeddings(token_ids, verbose=False, limit=None):
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

    # BERT 임베딩 파일을 numpy로 변환
    bert_embedding_dict_df = pd.read_csv('bert_embedding_dict.csv', index_col=0)
    bert_embedding_dict_np = np.array(bert_embedding_dict_df)
    
    # BERT 임베딩을 output data에 추가
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

        output_embedding = find_embedding_from_table(
            token=output_token,
            token_ids=token_ids,
            bert_embedding_dict_np=bert_embedding_dict_np,
            embed_limit=TKN_EMBEDDING_DIM
        )
        output_text_token_ids.append(output_embedding)

        train_data_iter_cnt += 1

        # flatten 해서 전체 데이터에 추가
        input_data_all.append(np.array(input_text_token_ids).flatten())
        output_data_all.append(np.array(output_text_token_ids).flatten())

    return np.array(input_data_all).astype(np.float16), np.array(output_data_all).astype(np.float16)


# 모델 학습 프로세스 전체 진행
def run_all_process(limit=None):
    global VOCAB_SIZE
    np.set_printoptions(linewidth=160)
    token_ids = get_token_ids()

    # vocab size 초기화
    VOCAB_SIZE = len(token_ids)
    print(f'VOCAB SIZE = {VOCAB_SIZE}')
    
    # 학습 데이터
    input_data_all, output_data_all = get_train_data_token_embeddings(token_ids, limit=limit)

    # 모델 학습 및 저장
    main_model = train_model(input_data_all, output_data_all)
    return main_model


# NLP 모델 테스트
def test_model(text, model, additional_tokenize=True, is_return=False, verbose=False, token_arr=None, token_ids=None, weight=None):
    if token_ids is None:
        token_ids = get_token_ids()

    if additional_tokenize:
        ing_map, ly_map = get_maps()
        tokenized_line = tokenize_line(text, ing_map, ly_map)
        tokenIzed_line_split = tokenized_line.split(' ')

        if len(tokenIzed_line_split) < INPUT_TOKEN_CNT_EACH:
            rest = INPUT_TOKEN_CNT_EACH - len(tokenIzed_line_split)
            tokenized_line = ('<null> ' * rest) + tokenized_line
            
        tokens = (tokenized_line.split(' ') + ['<null>'] * INPUT_TOKEN_CNT_EACH)
    else:
        tokens = text.split(' ')

    if verbose:
        print(f'\ntokens: {tokens}')
        print(f'original tokenized text: {text}')

    tokens_id = [token_ids[t] for t in tokens]

    if verbose:
        print(f'ID of tokens: {tokens_id}')

    output = model([tokens_id])

    if verbose:
        print(f'mini chatgpt model output: {output[0]}')

    if token_arr is None:
        token_arr = get_token_arr()

    if verbose:
        print(f'first 10 of token arr: {token_arr[:10]}')

    bert_embedding_table_df = pd.read_csv('bert_embedding_dict.csv', index_col=0)
    bert_embedding_table_np = np.array(bert_embedding_table_df)

    output_rank = find_nearest_bert_embedding_rank_for_embvec(
        embedding_vector=output[0].numpy(),
        bert_embedding_dict_np=bert_embedding_table_np,
        embed_limit=TKN_EMBEDDING_DIM,
        verbose=False,
        weight=weight
    )

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
    token_ids = get_token_ids()
    
    example_texts = [
        'what was the most number of people you have ever met during a working day ?',
        'i know him very well .',
        'how can i do for you ?',
        'how are you ?',
        'hello !'
    ]

    mini_chatgpt_model = tf.keras.models.load_model('mini_chatgpt_model')
    
    for example_text in example_texts:
        try:
            test_model(example_text, model=mini_chatgpt_model, verbose=True, token_ids=token_ids)
        except Exception as e:
            print(f'error: {e}')
