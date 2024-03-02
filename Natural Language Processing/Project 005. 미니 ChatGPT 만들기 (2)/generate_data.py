import numpy as np
import pandas as pd
from tokenize_data import tokenize_file_content


INPUT_SIZE = 36


# "Hi ! <person-change>" 체크
def is_hi(token_sequence):
    n = len(token_sequence)

    for i in range(n-2):
        if token_sequence[i] == 'hi' and token_sequence[i+1] == '!' and token_sequence[i+2] == '<person-change>':
            return True
    return False


# "<person-change> Hi" 체크
def is_hi_at_the_last(token_sequence):
    n = len(token_sequence)

    for i in range(n-1):
        if token_sequence[i] == '<person-change>' and token_sequence[i+1] == 'hi':
            return True
    return False


# 학습 데이터 생성
def generate_data(tokens, verbose=False):
    df = pd.DataFrame()

    for i in range(len(tokens) - INPUT_SIZE):
        if i % 750 == 0 and verbose:
            print(i)
            
        data_input = tokens[i:i + INPUT_SIZE + 1]
        data_row = ' '.join(data_input)

        # 입력 데이터 중 "Hi ! <Person-Change>" 가 있으면 그 전의 모든 token을 <Null> 로 교체한 새로운 row 추가
        data_input_except_first = tokens[i + 1:i + INPUT_SIZE]

        # 마지막 4개 token에서 "<Person-CHange> Hi" 가 발견되면 학습 데이터에서 제외
        if is_hi_at_the_last(data_input[-4:]):
            continue

        # 학습 데이터에 추가
        if is_hi(data_input_except_first):
            data_row_with_null = []
            is_hi_detected = False
            
            for j in range(INPUT_SIZE, -1, -1):
                if is_hi_detected:
                    data_row_with_null = ['<null>'] + data_row_with_null
                else:
                    data_row_with_null = [data_input[j]] + data_row_with_null

                    # check if 'Hi ! <Person-Change>' detected 
                    if len(data_input) >= j+3 and is_hi(data_input[j : j+3]):
                        is_hi_detected = True

            data_row_with_null = ' '.join(data_row_with_null)

            # <Null> 을 포함한 새로운 row 추가
            new_row_with_null = {'data': [data_row_with_null]}
            new_row_with_null = pd.DataFrame(new_row_with_null)
            df = pd.concat([df, new_row_with_null])

        else:
            # 새로운 row 추가
            new_row = {'data': [data_row]}
            new_row = pd.DataFrame(new_row)
            df = pd.concat([df, new_row])

    df.to_csv('train_data.csv')
    

if __name__ == '__main__':
    tokens = tokenize_file_content('archive/human_chat.txt')
    generate_data(tokens, verbose=True)
