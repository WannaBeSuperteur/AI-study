import pandas as pd
import numpy as np
import os

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, LeakyReLU, Dropout, Flatten, concatenate
from tensorflow.keras.utils import get_custom_objects

INPUT_TOKEN_CNT_EACH = 10 # 각 line 당 입력 토큰 개수
TKN_EMBEDDING_DIM = 32 # token embedding dimension
VOCAB_SIZE = None


# 사용자 정의 activation function = 1.5 * tanh(x) (-1 < tanh(x) < 1 인데, output 되는 값의 절댓값이 1보다 큰 경우 존재)
class Tanh_mul(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Tanh_mul, self).__init__(activation, **kwargs)
        self.__name__ = 'Tanh_mul'
        
def tanh_mul(x):
    return 1.5 * K.tanh(x)

get_custom_objects().update({'tanh_mul': Tanh_mul(tanh_mul)})


class MainModel(tf.keras.Model):
    
    def __init__(self, dropout_rate=0.25):
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
        self.BIDRC_LSTM_0 = Bidirectional(LSTM(units=64, dropout=0.3, return_state=True, return_sequences=True))
        self.BIDRC_LSTM_1 = Bidirectional(LSTM(units=128, dropout=0.3, return_state=True, return_sequences=True))
        self.BIDRC_LSTM_2 = Bidirectional(LSTM(units=128, dropout=0.3, return_state=True, return_sequences=True))
        self.BIDRC_LSTM_3 = Bidirectional(LSTM(units=64, dropout=0.3, return_state=True, return_sequences=True))

        # last turn, current turn Dense
        self.input_0_dense = Dense(64, activation=LeakyReLU(alpha=0.1))
        self.input_1_dense = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.input_2_dense = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.input_3_dense = Dense(64, activation=LeakyReLU(alpha=0.1))

        # dense layers
        self.dense0 = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.dense1 = Dense(32, activation='tanh_mul')
        self.final = Dense(1, activation='sigmoid')

        # flatten
        self.flatten = tf.keras.layers.Flatten()


    def call(self, inputs, training):
        input_0, input_1, input_2, input_3 = tf.split(
            inputs,
            [INPUT_TOKEN_CNT_EACH, INPUT_TOKEN_CNT_EACH, INPUT_TOKEN_CNT_EACH, INPUT_TOKEN_CNT_EACH],
            axis=1
        )

        # for input 0
        embed_tkn_0 = self.tkn_embedding(input_0)
        lstm_0, forward_h_0, forward_c_0, backward_h_0, backward_c_0 = self.BIDRC_LSTM_0(embed_tkn_0)
        lstm_0_concat = tf.keras.layers.Concatenate()([self.flatten(lstm_0), forward_h_0, forward_c_0, backward_h_0, backward_c_0])
        
        dense_0 = self.input_0_dense(lstm_0_concat)

        # for input 1
        embed_tkn_1 = self.tkn_embedding(input_1)
        lstm_1, forward_h_1, forward_c_1, backward_h_1, backward_c_1 = self.BIDRC_LSTM_1(embed_tkn_1)
        lstm_1_concat = tf.keras.layers.Concatenate()([self.flatten(lstm_1), forward_h_1, forward_c_1, backward_h_1, backward_c_1])
        
        dense_1 = self.input_1_dense(lstm_1_concat)

        # for input 2
        embed_tkn_2 = self.tkn_embedding(input_2)
        lstm_2, forward_h_2, forward_c_2, backward_h_2, backward_c_2 = self.BIDRC_LSTM_2(embed_tkn_2)
        lstm_2_concat = tf.keras.layers.Concatenate()([self.flatten(lstm_2), forward_h_2, forward_c_2, backward_h_2, backward_c_2])
        
        dense_2 = self.input_2_dense(lstm_2_concat)

        # for input 3
        embed_tkn_3 = self.tkn_embedding(input_3)
        lstm_3, forward_h_3, forward_c_3, backward_h_3, backward_c_3 = self.BIDRC_LSTM_3(embed_tkn_3)
        lstm_3_concat = tf.keras.layers.Concatenate()([self.flatten(lstm_3), forward_h_3, forward_c_3, backward_h_3, backward_c_3])
        
        dense_3 = self.input_3_dense(lstm_3_concat)

        # concatenation, ...
        all_concat = tf.keras.layers.Concatenate()([dense_0, dense_1, dense_2, dense_3])
        all_concat = self.dropout(all_concat)
        
        all_dense0 = self.dense0(all_concat)
        all_dense0 = self.dropout(all_dense0)
        all_dense1 = self.dense1(all_dense0)

        # final output
        outputs = self.final(all_dense1)
        return outputs


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, factor=0.125)
        
    model = MainModel()
    return model, optimizer, early_stopping, lr_reduced


# train, valid, test 데이터 나누기
def split_train_valid_test(df):
    df_len = len(df)
    data_np = np.array(df)

    train_np = data_np[:int(0.8 * df_len)]
    valid_np = data_np[int(0.8 * df_len):int(0.9 * df_len)]
    test_np = data_np[int(0.9 * df_len):]

    train_input = train_np[:, :-1].astype(np.int64)
    train_output = train_np[:, -1:]
    valid_input = valid_np[:, :-1].astype(np.int64)
    valid_output = valid_np[:, -1:]
    test_input = test_np[:, :-1].astype(np.int64)
    test_output = test_np[:, -1:]

    print(f'\ntrain input: {np.shape(train_input)}\n{train_input}')
    print(f'\ntrain output: {np.shape(train_output)}\n{train_output}')
    print(f'\nvalid input: {np.shape(valid_input)}\n{valid_input}')
    print(f'\nvalid output: {np.shape(valid_output)}\n{valid_output}')
    print(f'\ntest input: {np.shape(test_input)}\n{test_input}')
    print(f'\ntest output: {np.shape(test_output)}\n{test_output}')

    test_df = df.iloc[int(0.9 * df_len):]
    return train_input, train_output, valid_input, valid_output, test_input, test_output, test_df


# 모델 학습
def train_model(train_input, train_output, valid_input, valid_output):
    model, optimizer, early_stopping, lr_reduced = define_model()
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=80,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('main_model')

    return model


# VOCAB SIZE 계산
def compute_vocab_size():
    global VOCAB_SIZE
    
    data = pd.read_csv('train_data_token_id.csv', index_col=0)
    input_data = np.array(data)[:, :-1]
    VOCAB_SIZE = int(input_data.max() + 1)
    print(f'vocab size : {VOCAB_SIZE}')
    

if __name__ == '__main__':
    compute_vocab_size()

    # train, valid, test 데이터를 나누기 위해 shuffle 을 적용하여 데이터 로딩
    df = pd.read_csv('train_data_token_id.csv', index_col=0).sample(frac=1).reset_index(drop=True)

    # train, valid, test 데이터 나누기
    train_input, train_output, valid_input, valid_output, test_input, test_output, test_df = split_train_valid_test(df)

    # test data 저장하기
    test_df.to_csv('train_data_token_id_for_test.csv')

    # 학습 실시
    train_model(train_input, train_output, valid_input, valid_output)    
