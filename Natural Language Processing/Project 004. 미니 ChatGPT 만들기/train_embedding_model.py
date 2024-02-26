from embedding_helper import get_token_ids, encode_one_hot
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# fastest model from https://www.sbert.net/docs/pretrained_models.html
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
sbert_embeddings = {}

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 본 프로젝트에서 개발한 임베딩 모델
class EmbeddingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        self.encoder = tf.keras.Sequential([
            layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2),
            layers.Dense(units=16, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2),
            layers.Dense(units=384, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2)
        ])

    def call(self, inputs, training):
        embedding_result = self.encoder(inputs)
        outputs = self.decoder(embedding_result)
        return outputs


# S-BERT 로 임베딩
def embed_text_sbert(text):
    if text not in sbert_embeddings:
        sbert_embeddings[text] = sbert_model.encode(text)
    return sbert_embeddings[text]
        

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
        second_token_input = embed_text_sbert(tokens[1])
        first_token_output = embed_text_sbert(tokens[0])
        third_token_output = embed_text_sbert(tokens[2])
        
        input_data.append(second_token_input * A)
        output_data.append((first_token_output + third_token_output) / 2.0 * B)
        
        train_data_iter_cnt += 1

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    print(f'input data (shape: {np.shape(input_data)}) :\n{input_data}')
    print(f'output data (shape: {np.shape(output_data)}) :\n{output_data}')

    return input_data, output_data


# input data, output data 로부터 train, valid data 추출
def define_data(input_data, output_data, valid_ratio=0.1):
    valid_cnt = int(valid_ratio * len(input_data))
    
    train_input_data = input_data[:-valid_cnt]
    train_output_data = output_data[:-valid_cnt]

    valid_input_data = input_data[-valid_cnt:]
    valid_output_data = output_data[-valid_cnt:]

    return train_input_data, train_output_data, valid_input_data, valid_output_data


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = EmbeddingModel()
    return model, optimizer, early_stopping, lr_reduced


# 본 프로젝트에서 개발한 임베딩 모델 학습 진행 및 모델 저장
def train_model(train_input, train_output, valid_input, valid_output):
    model, optimizer, early_stopping, lr_reduced = define_model()
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=100,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('embedding_model')
    return model


# 본 프로젝트에서 개발한 임베딩 모델 테스트
def test_embedding_model(text):
    embedding_model = tf.keras.models.load_model('embedding_model')
    embedding_encoder = embedding_model.encoder

    if text not in sbert_embeddings:
        embed_text_sbert(text)
    
    sbert_embedding = sbert_embeddings[text]
    model_embedding = embedding_encoder(np.array([sbert_embedding]))
    print(f'{text} -> embedding:\n{np.array(model_embedding)}')
        

if __name__ == '__main__':
    input_data, output_data = get_train_data_as_embeddings(verbose=True)
    train_input, train_output, valid_input, valid_output = define_data(input_data, output_data)
    train_model(train_input, train_output, valid_input, valid_output)

    # 본 프로젝트에서 개발한 임베딩 모델 테스트
    test_tokens = 'you have to run .'
              
    for token in test_tokens.split(' '):
        test_embedding_model(token)
              
