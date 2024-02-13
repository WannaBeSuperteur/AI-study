import pandas as pd
import os
import numpy as np


window_size = 4
max_var_count = 12

vocab = [
    'if', 'print', 'elif', 'else', 'for', 'in', 'while',
    'def', 'return', '+', '-', '*', '/', '%', '=', '>', '<', ':',
    '[', ']', '{', '}', '(', ')', ',',
    'range', 'try', 'except', 'break', 'continue', '(n)', '(nl)',
    'True', 'False', 'and', 'or', 'not', 'class',
    'int', 'float', 'str', 'append', 'join', 'replace', 'split',
    'lambda', 'date', 'list', 'filter', 'split', 'set', 'dict',
    'numpy', 'np', 'pandas', 'pd', 'DataFrame', 'sum'
]

before_weight = [0.1, 0.2, 0.4, 0.7]
after_weight = before_weight[::-1]

assert len(before_weight) == window_size and len(after_weight) == window_size

for i in range(max_var_count):
    vocab.append('var' + str(i + 1))
    vocab.append('func' + str(i + 1))
    vocab.append('mytxttkn' + str(i + 1))

vocab_n = len(vocab)

vocab_dic = {}
for i in range(vocab_n):
    vocab_dic[vocab[i]] = i


import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K


# 임베딩 모델 (CBOW-like 이지만, 실제 CBOW 와 다소 차이가 있을 수 있음)
# input shape : (vocab_n)
# output shape : (vocab_n)
class EmbeddingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        self.encoder = tf.keras.Sequential([
            layers.Dense(units=64, activation='relu', kernel_regularizer=L2),
            layers.Dense(units=16, activation='relu', kernel_regularizer=L2)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(units=64, activation='relu', kernel_regularizer=L2),
            layers.Dense(units=1, activation='sigmoid', kernel_regularizer=L2)
        ])

    def call(self, inputs, training):
        embedding_result = self.encoder(inputs)
        outputs = self.decoder(embedding_result)
        return outputs


def get_vocab():
    return vocab


# 임베딩 결과 중 tokenize 된 코드 가져오기
def get_tokenized_codes():
    data_preprocessing_result = pd.read_csv('data_preprocessing_result.csv', index_col=0)
    return data_preprocessing_result['tokenized_code'].tolist()


# 토큰화된 Python code를 input (N) - output (1) - input (N) 형태로 변환
def convert_tokenized_code(tokenized_code):
    tokenized_code_split = tokenized_code.split(' ')
    
    code_len = len(tokenized_code_split)
    inputs_before = []
    outputs = []
    inputs_after = []
    
    for i in range(window_size, code_len - window_size):
        input_before = []
        output = []
        input_after = []
        
        for j in range(2 * window_size + 1):
            position = i + j - window_size
            
            if j < window_size:
                input_before.append(vocab_dic[tokenized_code_split[position]])
            elif j > window_size:
                input_after.append(vocab_dic[tokenized_code_split[position]])
            else:
                output.append(vocab_dic[tokenized_code_split[position]])

        inputs_before.append(input_before)
        outputs.append(output)
        inputs_after.append(input_after)

    # dataframe 형태로 만들어서 반환
    df = pd.DataFrame()

    for i in range(code_len - 2 * window_size): 

        new_row = {}
        for idx, ib in enumerate(inputs_before[i]):
            new_row[f'ib_{idx}'] = [int(inputs_before[i][idx])]

        new_row['out'] = [int(outputs[i][0])]

        for idx, ia in enumerate(inputs_after[i]):
            new_row[f'ia_{idx}'] = [int(inputs_after[i][idx])]

        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row])

    return df


# 데이터셋 생성 (CBOW 방식으로 word = token embedding 진행)
def create_dataset(tokenized_codes):
    embed_dataset = pd.DataFrame()
    df = pd.DataFrame()

    for idx, tokenized_code in enumerate(tokenized_codes):
        try:
            converted_code = convert_tokenized_code(tokenized_code)
            df = pd.concat([df, converted_code])
        except Exception as e:
            print(f'idx: {idx}, error: {e}, code: {str(tokenized_code)[:40]}')
    
    df.to_csv('embedding_dataset.csv')


# 임베딩 DataFrame -> One-Hot 으로 변환
def convert_into_one_hot(embedding_df):

    print('converting input-before ...')
    for i in range(window_size):
        for j in range(vocab_n):
            embedding_df[f'ib_{i}_{j}'] = embedding_df[f'ib_{i}'].apply(lambda x: 1 if x == j else 0)
        embedding_df.drop(columns=[f'ib_{i}'], inplace=True)

    print('converting output ...')
    for j in range(vocab_n):
        embedding_df[f'out_{j}'] = embedding_df['out'].apply(lambda x: 1 if x == j else 0)
    embedding_df.drop(columns=['out'], inplace=True)
    
    print('converting input-after ...')
    for i in range(window_size):
        for j in range(vocab_n):
            embedding_df[f'ia_{i}_{j}'] = embedding_df[f'ia_{i}'].apply(lambda x: 1 if x == j else 0)
        embedding_df.drop(columns=[f'ia_{i}'], inplace=True)


# ib_0 ~ ib_8 또는 ia_0 ~ ia_8 까지 평균 반환
def compute_all_window_mean(row, vocab_idx, before_after):
    result = 0

    for i in range(window_size):
        if before_after == 'before':
            result += before_weight[i] * row[f'ib_{i}_{vocab_idx}']
        elif before_after == 'after':
            result += after_weight[i] * row[f'ia_{i}_{vocab_idx}']
        
    return result


# 각 단어의 one-hot 벡터를 가중평균을 이용하여 합성 -> 최댓값으로 나누어서 최댓값이 1이 되게
def convert_to_mean(df):

    # position에 따른 가중평균을 이용하여 배열 계산
    for i in range(vocab_n):
        print(f'processing vocab index {i} (before) ...')
        df[f'token_before_{i}'] = df.apply(lambda x: compute_all_window_mean(x, i, before_after='before'), axis=1)

    for i in range(vocab_n):
        print(f'processing vocab index {i} (after) ...')
        df[f'token_after_{i}'] = df.apply(lambda x: compute_all_window_mean(x, i, before_after='after'), axis=1)

    # 불필요한 컬럼 삭제
    for i in range(window_size):
        print(f'removing columns for window-size {i} ...')
        
        for j in range(vocab_n):
            df.drop(columns=[f'ia_{i}_{j}'], inplace=True)
            df.drop(columns=[f'ib_{i}_{j}'], inplace=True)


# train, valid, test 데이터 반환
def define_data(train_data):
    train_n = len(train_data)
    token_cols = [f'token_before_{i}' for i in range(vocab_n)] + [f'token_after_{i}' for i in range(vocab_n)]
    out_cols_all = [f'out_{i}' for i in range(vocab_n)]

    mean_argmax = np.argmax(train_data[out_cols_all].mean())
    print('mean argmax:', mean_argmax)
    out_cols = [f'out_{mean_argmax}']

    train_input = np.array(train_data[token_cols], dtype=np.float32)
    train_output = np.array(train_data[out_cols], dtype=np.float32)

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
        
    model = EmbeddingModel()
    return model, optimizer, early_stopping, lr_reduced


# CBOW-like 학습 진행 및 모델 저장
def train_model(df):
    (train_input, valid_input, train_output, valid_output) = define_data(df)
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

    # test embedding model
    test_embedding_model(model)


# 임베딩 테스트 (모델 output을 출력)
def test_embedding_model(embedding_model):
    for i in range(15):
        test_arr = np.zeros(2 * vocab_n)
        test_arr[i] = 1
        test_arr[i + vocab_n] = 1
        
        test_arr = np.array([test_arr])
        print(f'\n{vocab[i]} -> {np.array(embedding_model(test_arr))}')


# 텍스트 임베딩 실시
def run_embedding(idx, emb_model_encoder):
    one_hot_arr = np.zeros((1, 2 * vocab_n))
    one_hot_arr[0][idx] = 1
    one_hot_arr[0][idx + vocab_n] = 1

    try:
        return emb_model_encoder(one_hot_arr)[0]
    except:
        one_hot_arr_tf = tf.convert_to_tensor(one_hot_arr, dtype=tf.float32, name='inputs')
        return emb_model_encoder(one_hot_arr_tf)[0]


# 임베딩 모델을 통한 CBOW 방식 학습
def create_data_for_cbow_like_model():

    df = pd.read_csv('embedding_dataset.csv', index_col=0)

    print('train token idx data:')
    print(df)

    convert_into_one_hot(df)

    print('\ntrain token idx data (modified 1):')
    print(df)

    convert_to_mean(df)

    print('\ntrain token idx data (modified 2):')
    print(df)

    df.to_csv('embedding_dataset_for_cbow.csv')


# 임베딩 모델을 통한 CBOW 방식 학습
def train_cbow_like_model():
    if 'embedding_dataset_for_cbow.csv' not in os.listdir():
        create_data_for_cbow_like_model()

    df = pd.read_csv('embedding_dataset_for_cbow.csv', index_col=0)
    train_model(df)


# 전체 프로세스 진행
def run_all_process():
    tokenized_codes = get_tokenized_codes()
    create_dataset(tokenized_codes)
    train_cbow_like_model()


if __name__ == '__main__':
    run_all_process()  
