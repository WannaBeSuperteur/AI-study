import pandas as pd


window_size = 9
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


# 임베딩 모델 (CBOW-like 이지만, 실제 CBOW 와 다소 차이가 있을 수 있음)
class EmbeddingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        L2 = tf.keras.regularizers.l2(0.001)

        self.embedding = layers.Dense(units=16, activation='sigmoid', kernel_regularizer=L2)
        self.output = layers.Dense(units=vocab_n, activation='sigmoid', kernel_regularizer=L2)

    def call(self, inputs, training):
        embedding = self.embedding(inputs)
        outputs = self.output(embedding)


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


# CBOW-like 학습 진행 및 모델 저장
def train_model(df):
    pass


# 임베딩 모델을 통한 CBOW 방식 학습
def train_cbow_like_model():
    df = pd.read_csv('embedding_dataset.csv', index_col=0)

    print('train token idx data:')
    print(df)

    convert_into_one_hot(df)

    print('\ntrain token idx data (modified):')
    print(df)

    print('\ncheck there are some 1s:')
    print(df.sum())

    train_model(df)


# 전체 프로세스 진행
def run_all_process():
    tokenized_codes = get_tokenized_codes()
    create_dataset(tokenized_codes)
    train_cbow_like_model()


if __name__ == '__main__':
    run_all_process()  
