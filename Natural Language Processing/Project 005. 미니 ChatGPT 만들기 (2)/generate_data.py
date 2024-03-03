import numpy as np
import pandas as pd
from tokenize_data import tokenize_file_content


INPUT_SIZE_EACH = 30


# <person-change> 토큰을 기준으로 대화 구분
def split_conversation(tokens):
    result = []
    current_turn = []

    for token in tokens:
        if token == '<person-change>':
            result.append(current_turn)
            current_turn = []

        else:
            current_turn.append(token)

    return result


# 학습 데이터 생성
def generate_data(tokens, verbose=False):
    df = pd.DataFrame()
    splitted_conversation = split_conversation(tokens)

    for i in range(len(splitted_conversation) - 1):
        if i % 125 == 0 and verbose:
            print(i)

        current_turn = splitted_conversation[i + 1]
        last_turn = splitted_conversation[i]

        # 이어지는 대화 턴이 "hi !", "hi ." 인 경우 제외
        if current_turn in [['hi', '!'], ['hi', '.']]:
            continue

        # 직전 턴
        if len(last_turn) < INPUT_SIZE_EACH:
            last_turn_rest = INPUT_SIZE_EACH - len(last_turn)
            last_turn = ['<null>'] * last_turn_rest + last_turn

        elif len(last_turn) > INPUT_SIZE_EACH:
            last_turn = last_turn[-INPUT_SIZE_EACH:]

        # 현재 턴
        for j in range(1, len(current_turn) + 1):
            current_turn_until_now = current_turn[:j]
            
            if len(current_turn_until_now) < INPUT_SIZE_EACH + 1:
                current_turn_rest = (INPUT_SIZE_EACH + 1) - len(current_turn_until_now)
                current_turn_until_now = ['<null>'] * current_turn_rest + current_turn_until_now

            elif len(current_turn_until_now) > INPUT_SIZE_EACH + 1:
                current_turn_until_now = current_turn_until_now[-(INPUT_SIZE_EACH + 1):]
                
            data_row = ' '.join(last_turn) + ' ' + ' '.join(current_turn_until_now)
            new_row = {'data': [data_row]}
            new_row = pd.DataFrame(new_row)
            df = pd.concat([df, new_row])

    df.to_csv('train_data.csv')
    

if __name__ == '__main__':
    tokens = tokenize_file_content('archive/human_chat.txt')
    generate_data(tokens, verbose=True)
