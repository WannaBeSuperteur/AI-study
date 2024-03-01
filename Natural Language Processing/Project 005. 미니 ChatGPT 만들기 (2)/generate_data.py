import numpy as np
import pandas as pd
from tokenize_data import tokenize_file_content


INPUT_SIZE = 24


# 학습 데이터 생성
def generate_data(tokens, verbose=False):
    df = pd.DataFrame()

    for i in range(len(tokens) - INPUT_SIZE):
        if i % 500 == 0 and verbose:
            print(i)
            
        data_input = tokens[i:i + INPUT_SIZE + 1]
        data_row = ' '.join(data_input)

        # 새로운 row 추가
        new_row = {'data': [data_row]}
        new_row = pd.DataFrame(new_row)
        df = pd.concat([df, new_row])

        # 입력 데이터 중 <Person-Change> 가 있으면 그 전의 모든 token을 <Null> 로 교체한 새로운 row 추가
        # 실제로는 입력 데이터의 최초 20개 token 중 <Person-Change> 가 있어야 함
        data_input_first_20 = tokens[i + 1:i + INPUT_SIZE - 4]
        
        if '<person-change>' in data_input_first_20:
            data_row_with_null = []
            is_person_change_detected = False
            
            for j in range(INPUT_SIZE, -1, -1):
                if is_person_change_detected:
                    data_row_with_null = ['<null>'] + data_row_with_null
                else:
                    data_row_with_null = [data_input[j]] + data_row_with_null
                    if data_input[j] == '<person-change>':
                        is_person_change_detected = True

            data_row_with_null = ' '.join(data_row_with_null)

            # <Null> 을 포함한 새로운 row 추가
            new_row_with_null = {'data': [data_row_with_null]}
            new_row_with_null = pd.DataFrame(new_row_with_null)
            df = pd.concat([df, new_row_with_null])

    df.to_csv('train_data.csv')
    

if __name__ == '__main__':
    tokens = tokenize_file_content('archive/human_chat.txt')
    generate_data(tokens, verbose=True)
