from embedding_helper import get_token_ids, load_embedding_encoder, get_embedding_of_token
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import get_custom_objects

from sentence_transformers import SentenceTransformer

# fastest model from https://www.sbert.net/docs/pretrained_models.html
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
sbert_embeddings = {}

embedding_dic = {}

INPUT_TOKEN_CNT = 16 # 학습 데이터 row 당 입력 토큰 개수
EMBEDDING_DIM = 24 # 본 프로젝트의 임베딩 모델에 의해 토큰이 임베딩 되는 dimension
SBERT_EMBED_SIZE_TO_USE = 128


# S-BERT 로 임베딩 (임베딩된 벡터의 384개의 값 중 처음 SBERT_EMBED_SIZE_TO_USE 개만 반환)
def embed_text_sbert(text):
    if text not in sbert_embeddings:
        sbert_embeddings[text] = sbert_model.encode(text)[:SBERT_EMBED_SIZE_TO_USE]
    return sbert_embeddings[text]


# latent vector model
class LatentVectorModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        # common NN
        self.common_NN = tf.keras.Sequential([
            layers.Dense(units=16, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='common_NN_0'),
            layers.Dense(units=4, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='common_NN_1')
        ])

        # inverse common NN
        self.inverse_common_NN = tf.keras.Sequential([
            layers.Dense(units=16, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='inverse_common_NN_0'),
            layers.Dense(units=24, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='inverse_common_NN_1')
        ])

        # encoder 부분 (4 dim x 16 tokens = 64 -> 32 -> 16)
        self.dense_0 = layers.Dense(units=32, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='dense_0')
        self.dense_1 = layers.Dense(units=16, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='dense_1')

        # decoder 부분 (16 -> 32 -> 64 = 4 dim x 16 tokens)
        self.dense_2 = layers.Dense(units=32, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='dense_2')
        self.dense_3 = layers.Dense(units=64, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=L2, name='dense_3')

    def call(self, inputs, training):
        input_length = inputs.shape[1] # 24 tokens x 16-dim vector for each tokens = 384
        token_cnt = input_length // EMBEDDING_DIM

        # encoding
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15 = tf.split(
            inputs, [EMBEDDING_DIM for i in range(token_cnt)], axis=1
        )

        tokens = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15]
        for i in range(len(tokens)):
            tokens[i] = self.common_NN(tokens[i])

        tokens_concat = tf.keras.layers.Concatenate()(tokens)
        tokens_concat = self.dense_0(tokens_concat)
        latent_vector = self.dense_1(tokens_concat)

        # decoding
        tokens_concat_ = self.dense_2(latent_vector)
        tokens_concat_ = self.dense_3(tokens_concat_)

        t0_, t1_, t2_, t3_, t4_, t5_, t6_, t7_, t8_, t9_, t10_, t11_, t12_, t13_, t14_, t15_ = tf.split(
            tokens_concat_, [4 for i in range(token_cnt)], axis=1
        )

        tokens_ = [t0_, t1_, t2_, t3_, t4_, t5_, t6_, t7_, t8_, t9_, t10_, t11_, t12_, t13_, t14_, t15_]
        for i in range(len(tokens)):
            tokens_[i] = self.inverse_common_NN(tokens_[i])

        # output
        outputs = tf.keras.layers.Concatenate()(tokens_)
        return outputs


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = LatentVectorModel()
    return model, optimizer, early_stopping, lr_reduced


# input data, output data 로부터 train, valid data 추출
def define_data(train_data_embedding, valid_ratio=0.1):
    
    train_data_embedding = np.reshape(train_data_embedding, (-1, INPUT_TOKEN_CNT * EMBEDDING_DIM))
    valid_cnt = int(valid_ratio * len(train_data_embedding))

    # define train and valid, input and output data
    train_input_data = train_data_embedding[:-valid_cnt]
    train_output_data = train_data_embedding[:-valid_cnt]

    valid_input_data = train_data_embedding[-valid_cnt:]
    valid_output_data = train_data_embedding[-valid_cnt:]

    # add random noise to input data
    train_random_noise = np.random.normal(0.0, 0.025, train_input_data.shape)
    train_input_data += train_random_noise

    valid_random_noise = np.random.normal(0.0, 0.025, valid_input_data.shape)
    valid_input_data += valid_random_noise

    return train_input_data, train_output_data, valid_input_data, valid_output_data


# 모델 학습 및 저장
def train_model(train_data_embedding):
    (train_input, train_output, valid_input, valid_output) = define_data(train_data_embedding)

    print(f'train input : {np.shape(train_input)}')
    print(f'valid input : {np.shape(valid_input)}')
    print(f'train output : {np.shape(train_output)}')
    print(f'valid output : {np.shape(valid_output)}')
    
    model, optimizer, early_stopping, lr_reduced = define_model()
    
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=10,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('latent_vector_model')

    return model


# latent vector 모델로부터 텍스트의 모든 token의 embedding의 concatenation에 대한 latent vector 추출
def get_latent_vector(text_embeddings, latent_vector_model):
    vectors_len_4 = [] # 4-dimension vectors each, for 16 tokens
    for embedding in text_embeddings:
        vectors_len_4.append(latent_vector_model.common_NN(embedding))
    vectors_len_4_concat = np.array(vectors_len_4).flatten() # 16 x 4 -> 64
    vectors_len_4_concat = np.array([vectors_len_4_concat])

    vectors_len_4_concat = latent_vector_model.dense_0(vectors_len_4_concat) # 32
    latent_vector = latent_vector_model.dense_1(vectors_len_4_concat) # 16

    return latent_vector


# 본 프로젝트에서 개발한 latent vector 모델 테스트
def test_latent_vector_model(text):
    embedding_model = tf.keras.models.load_model('embedding_model')
    latent_vector_model = tf.keras.models.load_model('latent_vector_model')
    embedding_encoder = embedding_model.encoder

    text_embeddings = []

    for token in text.split(' '):
        if token not in sbert_embeddings:
            embed_text_sbert(token)
        
        token_sbert_embedding = sbert_embeddings[token]
        token_embedding = embedding_encoder(np.array([token_sbert_embedding]))
        text_embeddings.append(token_embedding)
        print(f'\n{token} -> embedding:\n{np.array(token_embedding)}')

    # 텍스트 임베딩 concate 후 latent vector 추출
    print(f'\n{text} -> latent vector:\n{np.array(get_latent_vector(text_embeddings, latent_vector_model))}')


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
        
        tokens = row['data'].split(' ')[:input_size]

        train_data_one_hot_row = []
        for i in range(input_size):
            train_data_one_hot_row += list(embedding_dic[tokens[i]])
        train_data_one_hot.append(train_data_one_hot_row)

        train_data_iter_cnt += 1

    return np.array(train_data_one_hot)


if __name__ == '__main__':
    token_ids = get_token_ids()

    # 각 token을 본 프로젝트에서 개발한 임베딩 모델에 의한 embedding으로 매핑 
    fill_embedding_dic(token_ids)

    # 학습 데이터
    train_data_embedding = get_train_data_embedding(token_ids, verbose=True)

    # latent vector 모델 학습 및 저장
    latent_vector_model = train_model(train_data_embedding)

    # latent vector 모델 테스트 (each example text has 16 tokens)
    example_texts = [
        'what was the most number of people you have ever met during a working day ?',
        'i know him very well . <Person-Change> is him your friend ? if so , it',
        'how can i do for you ? <Person-Change> can you borrow me a science book ?'
    ]
    
    for example_text in example_texts:
        test_latent_vector_model(example_text)
    
    print(f'shape of train_data_embedding: {np.shape(train_data_embedding)}')
