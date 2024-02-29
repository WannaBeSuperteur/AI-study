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


INPUT_TOKEN_CNT = 16 # 학습 데이터 row 당 입력 토큰 개수
EMBEDDING_DIM = 24 # 본 프로젝트의 임베딩 모델에 의해 토큰이 임베딩 되는 dimension
SBERT_EMBED_SIZE_TO_USE = 128 # S-BERT embedding 값 384개 중 처음 128개만 이용
MULTIPLE = 2.5 # Normal Distribution에 가까워지도록 input, output data에 일정한 값을 곱하여 학습


# 사용자 정의 activation function = 1.5 * tanh(x) (-1 < tanh(x) < 1 인데, output 되는 값의 절댓값이 1보다 큰 경우 존재)
class Tanh_mul(tf.keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super(Tanh_mul, self).__init__(activation, **kwargs)
        self.__name__ = 'Tanh_mul'
        
def tanh_mul(x):
    return 1.5 * K.tanh(x)

get_custom_objects().update({'tanh_mul': Tanh_mul(tanh_mul)})


# S-BERT 로 임베딩 (임베딩된 벡터의 384개의 값 중 처음 SBERT_EMBED_SIZE_TO_USE 개만 반환)
def embed_text_sbert(text):
    if text not in sbert_embeddings:
        sbert_embeddings[text] = sbert_model.encode(text)[:SBERT_EMBED_SIZE_TO_USE]
    return sbert_embeddings[text]


# main model
class MainModel(tf.keras.Model):
    
    def __init__(self, dropout_rate=0.25):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        self.dense_0 = layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2)
        self.dense_1 = layers.Dense(units=256, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2)
        self.dense_2 = layers.Dense(units=128, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2)
        self.dense_3 = layers.Dense(units=24, activation='tanh_mul', kernel_regularizer=L2)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')

    def call(self, inputs, training):
        inputs = self.dense_0(inputs)
        inputs = self.dropout(inputs)
        inputs = self.dense_1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.dense_2(inputs)
        inputs = self.dropout(inputs)
        
        outputs = self.dense_3(inputs)
        return outputs


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = MainModel()
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


# 모델 학습 및 저장
def train_model(input_data_all, output_data_all):
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
        epochs=50,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('main_model')

    return model


# 본 프로젝트에서 만든 임베딩 모델에 의한 embedding
def get_embedding(token, embedding_encoder):
    if token not in sbert_embeddings:
        embed_text_sbert(token)
            
    token_sbert_embedding = sbert_embeddings[token]
    token_embedding = embedding_encoder(np.array([token_sbert_embedding]))
    return token_embedding.numpy()[0].tolist()


# train_data.csv 파일의 데이터 전체를,
# 본 프로젝트에서 개발한 임베딩 모델에 의한 임베딩 배열로 만들어서 반환
def get_train_data_embedding(token_ids, verbose=False, limit=None):
    embedding_model = tf.keras.models.load_model('embedding_model')
    embedding_encoder = embedding_model.encoder
    
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
        if train_data_iter_cnt % 100 == 0:
            print(f'count: {train_data_iter_cnt}')

        if limit is not None and train_data_iter_cnt >= limit:
            break
        
        input_tokens = row['data'].split(' ')[:input_size]
        output_token = row['data'].split(' ')[-1]
        input_text_embeddings = []
        output_text_embeddings = []

        for input_token in input_tokens:
            input_token_embedding = get_embedding(input_token, embedding_encoder)
            input_text_embeddings.append(input_token_embedding)

        output_token_embedding = get_embedding(output_token, embedding_encoder)
        output_text_embeddings.append(output_token_embedding)

        train_data_iter_cnt += 1

        # flatten 해서 전체 데이터에 추가
        input_data_all.append(np.array(input_text_embeddings).flatten())
        output_data_all.append(np.array(output_text_embeddings).flatten())

    input_data_all = np.array(input_data_all) * MULTIPLE
    output_data_all = np.array(output_data_all) * MULTIPLE

    return input_data_all, output_data_all


# 본 프로젝트에서 개발한 메인 모델 테스트
def test_main_model(text):
    embedding_model = tf.keras.models.load_model('embedding_model')
    embedding_encoder = embedding_model.encoder

    main_model = tf.keras.models.load_model('main_model')

    text_embedding = []

    for token in text.split(' '):
        token_embedding = get_embedding(token, embedding_encoder)
        text_embedding += token_embedding

    # 텍스트 임베딩 concate 후 main model 출력 결과 추출
    # 실제로 가장 가까운 token embedding 을 계산할 때는 해당 출력 결과를 먼저 MULTIPLE 로 나누어서,
    # 원래 값을 복구한 후 그 복구한 값을 이용하여 찾아야 함
    print(f'\n{text} -> main model result:\n{np.array(main_model([text_embedding]))}')


# 메인 모델 학습 프로세스 전체 진행
def run_all_process(limit=None):
    token_ids = get_token_ids()
    
    # 학습 데이터
    input_data_all, output_data_all = get_train_data_embedding(token_ids, limit=limit)

    # 메인 모델 학습 및 저장
    main_model = train_model(input_data_all, output_data_all)
    return main_model


if __name__ == '__main__':
    run_all_process()

    # 메인 모델 테스트 (each example text has 16 tokens)
    example_texts = [
        'what was the most number of people you have ever met during a working day ?',
        'i know him very well . <Person-Change> is him your friend ? if so , it',
        'how can i do for you ? <Person-Change> can you borrow me a science book ?'
    ]
    
    for example_text in example_texts:
        test_main_model(example_text)
