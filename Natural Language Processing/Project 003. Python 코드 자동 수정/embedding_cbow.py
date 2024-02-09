import pandas as pd


window_size = 6
max_var_count = 30

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
    vocab.append('var' + str(i))
    vocab.append('func' + str(i))
    vocab.append('mytxttkn' + str(i))

vocab_n = len(vocab)

vocab_dic = {}
for i in range(vocab_n):
    vocab_dic[vocab[i]] = i


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
            if j < window_size:
                input_before.append(vocab_dic[tokenized_code_split[j]])
            elif j > window_size:
                input_after.append(vocab_dic[tokenized_code_split[j]])
            else:
                output.append(vocab_dic[tokenized_code_split[j]])

        inputs_before.append(input_before)
        outputs.append(output)
        inputs_after.append(input_after)

    # dataframe 형태로 만들어서 반환
    df = pd.DataFrame()

    for i in range(code_len - 2 * window_size): 

        new_row = {}
        for idx, ib in enumerate(inputs_before):
            new_row[f'ib_{idx}'] = [int(inputs_before[i][idx])]

        new_row['out'] = [int(outputs[i][0])]

        for idx, ia in enumerate(inputs_after):
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


# 임베딩 모델을 통한 CBOW 방식 학습
def train_cbow_model():
    pass


if __name__ == '__main__':
    tokenized_codes = get_tokenized_codes()
    print(tokenized_codes[0])
    create_dataset(tokenized_codes)    
