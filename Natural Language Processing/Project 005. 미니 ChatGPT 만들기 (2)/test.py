from embedding_helper import get_token_ids
from tokenize_data import tokenize_line, get_maps
from train import test_model
import math
import tensorflow as tf
import numpy as np

NUM_INPUT_TOKENS = 36
ing_map, ly_map = get_maps()


# 사용자 입력을 tokenize 하는 함수
def tokenize_for_test(text):
    text = tokenize_line(text, ing_map, ly_map)

    text_tokens = len(text.split(' '))

    if text_tokens > NUM_INPUT_TOKENS - 1:
        text = ' '.join(text.split(' ')[-(NUM_INPUT_TOKENS - 1):])
    
    if text_tokens < NUM_INPUT_TOKENS - 1:
        rest = (NUM_INPUT_TOKENS - 1) - text_tokens
        text = '<null> ' * rest + text
        
    return text + ' <person-change>'


# 사용자 입력
def get_user_input():
    input_text = input('\ninput text :\n')
    return tokenize_for_test(input_text)


if __name__ == '__main__':
    mini_chatgpt_model = tf.keras.models.load_model('mini_chatgpt_model')
    token_ids = get_token_ids()
    weight = np.random.uniform(0.5, 1.5, NUM_INPUT_TOKENS)

    # 사용자 입력
    tokenized_input = get_user_input()
        
    outputs = []

    for i in range(12):
        print(f'tokenized input : {tokenized_input}')
        
        # 다음 토큰 예측
        try:
            next_output_rank = test_model(tokenized_input, model=mini_chatgpt_model, additional_tokenize=False, is_return=True)
            next_output = next_output_rank[0][0]

            for j in range(10):
                print(next_output_rank[j])

            outputs.append(next_output)
            tokenized_input = ' '.join(tokenized_input.split(' ')[1:]) + ' ' + next_output

        except Exception as e:
            if 'KeyError:' in str(e):
                print('다음 단어가 사전에 없습니다:', e.split("'")[1])
            else:
                print(f'error: {e}')

    print('outputs:', ' '.join(outputs))
