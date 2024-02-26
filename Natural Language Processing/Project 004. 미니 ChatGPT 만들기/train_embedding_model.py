from embedding_helper import get_token_ids, encode_one_hot
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# fastest model from https://www.sbert.net/docs/pretrained_models.html
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
sbert_embeddings = {}

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import get_custom_objects

SBERT_EMBED_SIZE_TO_USE = 128 # S-BERT embedding 값 384개 중 128개만 이용
DEST_EMBED_SIZE = 24


# 사용자 정의 activation function = 1.25 * tanh(x) (기존 tanh는 미분값 평균 1 미만으로 Vanishing Gradient Problem 존재) + Zero-centered
class Tanh_mul(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Tanh_mul, self).__init__(activation, **kwargs)
        self.__name__ = 'Tanh_mul'
        
def tanh_mul(x):
    return 1.25 * K.tanh(x)

get_custom_objects().update({'tanh_mul': Tanh_mul(tanh_mul)})



# 본 프로젝트에서 개발한 임베딩 모델
# tf.keras.layers.LeakyReLU(alpha=0.25) or tanh?
class EmbeddingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        self.encoder = tf.keras.Sequential([
            layers.Dense(units=512, activation='tanh_mul', kernel_regularizer=L2),
            layers.Dense(units=64, activation='tanh_mul', kernel_regularizer=L2),
            layers.Dense(units=DEST_EMBED_SIZE, activation='tanh_mul', kernel_regularizer=L2)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(units=64, activation='tanh_mul', kernel_regularizer=L2),
            layers.Dense(units=512, activation='tanh_mul', kernel_regularizer=L2),
            layers.Dense(units=SBERT_EMBED_SIZE_TO_USE, activation='tanh_mul', kernel_regularizer=L2)
        ])

    def call(self, inputs, training):
        embedding_result = self.encoder(inputs)
        outputs = self.decoder(embedding_result)
        return outputs


# S-BERT 로 임베딩 (임베딩된 벡터의 384개의 값 중 처음 SBERT_EMBED_SIZE_TO_USE 개만 반환)
def embed_text_sbert(text):
    if text not in sbert_embeddings:
        sbert_embeddings[text] = sbert_model.encode(text)[:SBERT_EMBED_SIZE_TO_USE]
    return sbert_embeddings[text]
        

# train_data.csv 파일의 데이터를 S-BERT 임베딩으로 변환해서 반환
# 입력 : (2번째 단어의 S-BERT embedding vector) * A + (small random noise)
# 출력 : (입력과 동일한 2번째 단어의 S-BERT embedding vector) * A
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

    # embedding을 "표준정규분포에 가까운" input, output data 로 변환하기 위해 곱하는 값
    A = 1.2

    for _, row in train_data.iterrows():
        if train_data_iter_cnt % 1500 == 0:
            print(f'count: {train_data_iter_cnt}')
        
        tokens = row['data'].split(' ')
        second_token_input = embed_text_sbert(tokens[1])
        second_token_output = embed_text_sbert(tokens[1])

        random_noise = 0.05 * np.random.randn(SBERT_EMBED_SIZE_TO_USE)
        input_data.append(second_token_input * A + random_noise)
        output_data.append(second_token_input * A)
        
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
    print(f'\n{text} -> embedding:\n{np.array(model_embedding)}')
        

if __name__ == '__main__':
    input_data, output_data = get_train_data_as_embeddings(verbose=True)
    train_input, train_output, valid_input, valid_output = define_data(input_data, output_data)
    train_model(train_input, train_output, valid_input, valid_output)

    # 본 프로젝트에서 개발한 임베딩 모델 테스트
    test_tokens = 'you have to run . good great best awesome bad worse worst'
              
    for token in test_tokens.split(' '):
        test_embedding_model(token)
              
