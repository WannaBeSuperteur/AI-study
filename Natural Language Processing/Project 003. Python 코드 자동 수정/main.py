import read_data as rd
import tokenizer
import pandas as pd
import embedding_cbow as ecbow
from embedding_cbow import get_vocab
import os
from generate_dataset import get_or_create_main_model_train_data

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np


window_size = 4
embedding_size = 16 # embedding vector dimension
vocab_n = 94 # embedding_cbow.py 의 vocab 구조 변경 시 업데이트


class MainModel(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        self.dense0 = layers.Dense(units=256, activation='relu', kernel_regularizer=L2)
        self.dense1 = layers.Dense(units=512, activation='relu', kernel_regularizer=L2)
        self.dense2 = layers.Dense(units=64, activation='relu', kernel_regularizer=L2)
        self.dense3 = layers.Dense(units=7, activation='sigmoid', kernel_regularizer=L2)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')

    def call(self, inputs, training):
        inputs = self.dense0(inputs)
        inputs = self.dropout(inputs)
        inputs = self.dense1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.dense2(inputs)
        inputs = self.dropout(inputs)
        
        outputs = self.dense3(inputs)
        return outputs


# train, valid, test 데이터 반환
def define_data(train_df):
    train_n = len(train_df)
    train_df = train_df.sample(frac=1.0).reset_index(drop=True) # shuffle train_df
    print(train_df)
    
    embedding_element_cols = []
    for i in range(window_size):
        for j in range(embedding_size):
            embedding_element_cols.append(f'ia_{i}_{j}')
            embedding_element_cols.append(f'ib_{i}_{j}')

    # 각 out column의 데이터의 평균 파악
    vocab = get_vocab()
    for i in range(vocab_n):
        mean_value = np.mean(train_df[f'out_{i}'])
        if mean_value >= 0.05:
            print(f'mean of column out_{i} ({vocab[i]}) : ' + str(mean_value))
        
    # out_cols = [f'out_{i}' for i in range(vocab_n)]

    # 평균으로 수렴하는 것을 방지하기 위해,
    # one-hot vector 기준 평균값이 0.05 이상인 column ('=', ':', '(', ')', ',', '(n)', '(nl)') 만 이용
    out_cols = ['out_14', 'out_17', 'out_22', 'out_23', 'out_24', 'out_30', 'out_31']

    train_input = np.array(train_df[embedding_element_cols], dtype=np.float32)
    train_output = np.array(train_df[out_cols], dtype=np.float32)

    print(train_input)
    print(np.shape(train_input))
    
    print(train_output)
    print(np.shape(train_output))

    valid_count = int(0.2 * train_n)
    train_input_train = train_input[:-valid_count]
    train_input_valid = train_input[-valid_count:]
    train_output_train = train_output[:-valid_count]
    train_output_valid = train_output[-valid_count:]

    print('train_input - train shape :', np.shape(train_input_train))
    print('train_input - valid shape :', np.shape(train_input_valid))
    print('train_output - train shape :', np.shape(train_output_train))
    print('train_output - valid shape :', np.shape(train_output_valid))

    return (train_input_train, train_input_valid, train_output_train, train_output_valid)


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = MainModel(dropout_rate=0.25)
    return model, optimizer, early_stopping, lr_reduced


# 테스트 (모델 output을 출력 -> 평균으로 수렴하지 않았는지 테스트)
def test_main_model(main_model):
    for i in range(15):
        test_arr = np.random.rand(1, 2 * window_size * embedding_size)
        print(f'random array {i + 1} -> {np.array(main_model(test_arr))}')


# 메인 모델 학습
def train_main_model(main_model_df):
    (train_input, valid_input, train_output, valid_output) = define_data(main_model_df)

    model, optimizer, early_stopping, lr_reduced = define_model()
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=100,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('main_model')
    test_main_model(model)


# tokenize 하고, 그 텍스트를 공백으로 연결하여 반환
def tokenize_with_join(text):
    try:
        text = tokenizer.tokenize(text)
        return ' '.join(text)
    
    except:
        return None


# tokenize
def tokenize_converted_data():
    df = pd.read_csv('converted_data.csv', index_col=0)
    df['tokenized_code'] = df['code'].apply(lambda x: tokenize_with_join(x))
    df.to_csv('data_preprocessing_result.csv')


# Embedding Model 용으로, 데이터 읽고 model train 용 데이터 생성까지 진행
def generate_model_train_data():
    
    # Python code 원본 데이터 읽기
    rd.run_convert()

    # tokenize 실시
    tokenize_converted_data()

    # CBOW embedding 학습 실시
    ecbow.run_all_process()
    

if __name__ == '__main__':
    if 'embedding_dataset_for_cbow.csv' not in os.listdir():
        print('no embedding dataset for CBOW-like embedding model')
        generate_model_train_data()
        
    elif 'embedding_model' not in os.listdir():
        print('no embedding model')
        ecbow.train_cbow_like_model()

    else:
        print('embedding model prepared')

    # 메인 모델용 학습 데이터 로딩
    main_model_df = get_or_create_main_model_train_data()
    print(main_model_df)

    # 메인 모델 학습
    train_main_model(main_model_df)
