import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import math

from preprocess_data import read_data, add_age_of_user_numeric_col
import nltk
from sentence_transformers import SentenceTransformer

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# fastest model from https://www.sbert.net/docs/pretrained_models.html
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
EMBEDDING_DIM = 384

class SentimentModel(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        self.dense_info0 = layers.Dense(units=512, activation='relu', kernel_regularizer=L2)
        self.dense_info1 = layers.Dense(units=256, activation='relu', kernel_regularizer=L2)
        self.dense_info2 = layers.Dense(units=16, activation='relu', kernel_regularizer=L2)

        self.dense_embed0 = layers.Dense(units=16, activation='relu', kernel_regularizer=L2)
        
        self.dense_final0 = layers.Dense(units=16, activation='sigmoid', kernel_regularizer=L2)
        self.dense_final1 = layers.Dense(units=1, activation='sigmoid', kernel_regularizer=L2)
        
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training):
        input_length = inputs.shape[1]
        inputs_info, inputs_emb = tf.split(inputs, [input_length - EMBEDDING_DIM, EMBEDDING_DIM], axis=1)
        
        inputs_info = self.dense_info0(inputs_info)
        inputs_info = self.dropout(inputs_info)
        inputs_info = self.dense_info1(inputs_info)
        inputs_info = self.dropout(inputs_info)
        inputs_info = self.dense_info2(inputs_info)
        inputs_info = self.dropout(inputs_info)

        inputs_emb = self.dense_embed0(inputs_emb)
        inputs_emb = self.dropout(inputs_emb)

        inputs_concat = tf.keras.layers.Concatenate()([inputs_info, inputs_emb])
        inputs_concat = self.dense_final0(inputs_concat)
        outputs = self.dense_final1(inputs_concat)

        return outputs


def load_data():
    train_data = read_data('train.csv')
    test_data = read_data('test.csv')

    add_age_of_user_numeric_col(train_data)
    add_age_of_user_numeric_col(test_data)

    return train_data, test_data


def remove_unused_columns(train_data, test_data):
    train_data.drop(columns=['textID', 'selected_text', 'Age of User'], inplace=True)
    test_data.drop(columns=['textID', 'Age of User'], inplace=True)


def make_one_hot_columns(data, columns, base_data):
    for column in columns:
        vcs_top20 = base_data[column].value_counts().head(20)
        
        unique_vals = list(vcs_top20.index)
        unique_vals.sort() # train, test에서 one-hot item 순서가 고정되게

        for unique_val in unique_vals:
            new_col = column + '_' + unique_val
            data[new_col] = data[column].apply(
                lambda x: 1 if x == unique_val else 0
            )

        data.drop(columns=[column], inplace=True)


def apply_normalize(data, column, apply_log=False):
    if apply_log:
        data[column] = data[column].apply(lambda x: np.log(x + 1))
    
    mean = data[column].mean()
    std = data[column].std()

    data[column] = data[column].apply(lambda x: (x - mean) / std)


def remove_website_addresses(text):
    words = text.split(' ')
    words = list(filter(lambda x: 'http://' not in x and 'https://' not in x, words))
    return ' '.join(words)


def preprocess_text(text):
    text = remove_website_addresses(text)
    
    tokens = nltk.word_tokenize(text)
    tags = nltk.tag.pos_tag(tokens)

    tags = list(filter(lambda x: x[1] not in ['(', ')', ':'], tags))
    tags = [tag[0] for tag in tags]
    preprocessed_text = ' '.join(tags)

    return preprocessed_text


def apply_preprocess_text(data):
    data['text'] = data['text'].apply(lambda x: preprocess_text(x))


def preprocess_before_training(train_data, test_data):
    add_age_of_user_numeric_col(train_data)
    add_age_of_user_numeric_col(test_data)

    # 불필요한 컬럼 제거
    remove_unused_columns(train_data, test_data)

    # one-hot 처리
    one_hot_cols = ['Time of Tweet', 'Country']
    train_data_copy = pd.DataFrame(train_data)
    
    make_one_hot_columns(train_data, one_hot_cols, train_data_copy)
    make_one_hot_columns(test_data, one_hot_cols, train_data_copy)

    # normalize 처리
    numeric_cols = ['Age of User Numeric']
    numeric_log_cols = ['Population -2020', 'Land Area (Km)', 'Density (P/Km)']

    for numeric_col in numeric_cols:
        apply_normalize(train_data, numeric_col, apply_log=False)
        apply_normalize(test_data, numeric_col, apply_log=False)

    for numeric_log_col in numeric_log_cols:
        apply_normalize(train_data, numeric_log_col, apply_log=True)
        apply_normalize(test_data, numeric_log_col, apply_log=True)

    # target column (sentiment) 처리
    sentiment_mapping = {'positive': 1.0, 'neutral': 0.5, 'negative': 0.0}
    train_data['sentiment'] = train_data['sentiment'].apply(lambda x: sentiment_mapping[x])
    test_data['sentiment'] = test_data['sentiment'].apply(lambda x: sentiment_mapping[x])


def create_sbert_embeddings(data):
    embeddings = []

    for idx, row in data.iterrows():
        if idx % 1000 == 0:
            print(idx)
            
        embedding = sbert_model.encode(row['text'])
        embeddings.append(embedding)

    return embeddings


def apply_sbert(data):
    embeddings = create_sbert_embeddings(data)
    embedding_dim = len(embeddings[0])
    
    for emb in range(embedding_dim):
        data[f'emb_{emb}'] = data.index.map(lambda x: embeddings[x][emb])
    data.drop(columns=['text'], inplace=True)


def preprocess_text_in_data(train_data, test_data):
    apply_preprocess_text(train_data)
    apply_preprocess_text(test_data)

    print('after text preprocessing:')
    print(train_data)
    print(test_data)

    apply_sbert(train_data)
    apply_sbert(test_data)


# train, valid, test 데이터 반환
def define_data(train_data, test_data):
    train_n = len(train_data)

    train_input = train_data.drop(columns=['sentiment'])
    train_output = train_data[['sentiment']]

    print(train_input)
    print(train_output)

    train_input = np.array(train_input)
    train_output = np.array(train_output)

    valid_count = int(0.2 * train_n)
    train_input_train = train_input[:-valid_count]
    train_input_valid = train_input[-valid_count:]
    train_output_train = train_output[:-valid_count]
    train_output_valid = train_output[-valid_count:]

    test_input = test_data.drop(columns=['sentiment'])
    test_gt = test_data[['sentiment']]

    print(test_input)
    print(test_gt)

    return (train_input_train, train_input_valid, train_output_train,
            train_output_valid, test_input, test_gt)
           

# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = SentimentModel(dropout_rate=0.25)
    return model, optimizer, early_stopping, lr_reduced


# 모델 학습 및 저장
def train_model(train_data, test_data):
    (train_input, valid_input, train_output, valid_output, test_input, test_gt) = define_data(train_data, test_data)
    model, optimizer, early_stopping, lr_reduced = define_model()
    
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=10,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('sentiment_model')


if __name__ == '__main__':
    train_data, test_data = load_data()

    # there is a row index jump near train_data[315]
    train_data.index = pd.RangeIndex(len(train_data.index))
    test_data.index = pd.RangeIndex(len(test_data.index))
    
    print(train_data)
    print(test_data)

    preprocess_before_training(train_data, test_data)
    preprocess_text_in_data(train_data, test_data)

    print('after text embedding:')
    print(train_data)
    print(test_data)

    train_data.to_csv('train_data_final.csv')
    test_data.to_csv('test_data_final.csv')

    # train model
    train_model(train_data, test_data)
