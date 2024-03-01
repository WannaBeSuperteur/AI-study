import numpy as np
import pandas as pd
from tokenize_data import tokenize_file_content


INPUT_SIZE = 16


# 학습 데이터 생성
def generate_data(tokens):
    df = pd.DataFrame()

    for i in range(len(tokens) - INPUT_SIZE):
        data_row = ' '.join(tokens[i:i + INPUT_SIZE + 1])
        new_row = {'data': [data_row]}
        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row])

    df.to_csv('train_data.csv')
    

if __name__ == '__main__':
    tokens = tokenize_file_content('archive/human_chat.txt')
    generate_data(tokens)
