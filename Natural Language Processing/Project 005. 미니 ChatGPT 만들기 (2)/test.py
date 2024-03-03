from embedding_helper import get_token_ids, get_token_arr, validate_embedding, print_most_similar_tokens
from tokenize_data import tokenize_line, get_maps
from train import test_model
from train import INPUT_TOKEN_CNT_EACH, TKN_EMBEDDING_DIM

from add_bert_embedding_dict import find_nearest_bert_embedding, find_nearest_bert_embedding_rank

import math
import tensorflow as tf
import numpy as np
import pandas as pd
import random

ing_map, ly_map = get_maps()
token_ids = get_token_ids()


# vocab에 없는 토큰 처리 함수
def handle_tokens_not_in_vocab(text):
    text_split = text.split(' ')

    for i in range(len(text_split)):
        if text_split[i] not in token_ids:
            replaced = find_nearest_bert_embedding(text_split[i])
            print(f'{text_split[i]} 는 vocab 에 없으므로 {replaced} 로 대체됨')
            text_split[i] = replaced

    return ' '.join(text_split)


# 사용자 입력을 tokenize 하는 함수
def tokenize_for_test(text, fill_rest_null=True):
    text = tokenize_line(text, ing_map, ly_map)

    text_tokens = len(text.split(' '))

    if text_tokens > INPUT_TOKEN_CNT_EACH:
        text = ' '.join(text.split(' ')[-INPUT_TOKEN_CNT_EACH:])
    
    if fill_rest_null and text_tokens < INPUT_TOKEN_CNT_EACH:
        rest = INPUT_TOKEN_CNT_EACH - text_tokens
        text = '<null> ' * rest + text

    # vocab에 없는 토큰 처리 후 반환
    text = handle_tokens_not_in_vocab(text)
    return text


# 사용자 입력
def get_user_input(fill_rest_null=True):
    input_text = input('\ninput text :\n')
    return tokenize_for_test(input_text, fill_rest_null=fill_rest_null)


# 생성형 출력을 위해, embedding array A 저장
def get_embedding_array_A(mini_chatgpt_model):
    vocab_size = len(token_ids)
    A = np.zeros((vocab_size, TKN_EMBEDDING_DIM))

    for token, idx in token_ids.items():
        embedding = mini_chatgpt_model.tkn_embedding(np.array([idx]).astype(np.int32))
        embedding = embedding.numpy()[0]
        A[idx] = embedding

    return A


# next_output_rank (tokne의 출력값 순위 정보) 재정렬
def restore_next_output_rank(next_output_rank, A, B):
    vocab_size = len(token_ids)

    for i in range(vocab_size):
        token = next_output_rank[i][0]
        vocab_idx = token_ids[token]

        AB_sum_product = sum([abs(A[vocab_idx][d]) * B[d] for d in range(TKN_EMBEDDING_DIM)])
        next_output_rank[i][1] = next_output_rank[i][1] * AB_sum_product

    next_output_rank.sort(key=lambda x:x[1], reverse=True)


# 생성형 AI로 만들기 위해, 다음 토큰을 확률적으로 선택 (단, threshold는 token 별 weight의 최댓값의 50% 이하만 가능)
def choose_one_token(next_output_rank, threshold=0.3, verbose=False):
    candidates = []
    candidate_probs = []

    max_weight = max([rank[1] for rank in next_output_rank])
    threshold = min(threshold, 0.5 * max_weight)

    if verbose:
        print(f'maximum weight:\n{max_weight}')
    
    for i in range(len(next_output_rank)):
        token = next_output_rank[i][0]
        weight = next_output_rank[i][1]
        
        if weight >= threshold:
            candidates.append(token)
            candidate_probs.append(weight)

    if verbose:
        print(f'candidates:\n{candidates}')
        print(f'candidate probs:\n{candidate_probs}')

    choice = random.choices(candidates, weights=candidate_probs)[0]
    return choice
    

if __name__ == '__main__':
    mini_chatgpt_model = tf.keras.models.load_model('mini_chatgpt_model')
    token_arr = get_token_arr()

    # next output rank를 생성형 예측으로 만들기 위한 embedding layer A와 random init 배열 B
    A = get_embedding_array_A(mini_chatgpt_model)
    B = np.random.uniform(0.5, 1.5, TKN_EMBEDDING_DIM)

    print(f'embedding array A: ({np.shape(A)})\n{A}\n')
    print(f'random array B: ({np.shape(B)})\n{B}\n')

    # 임베딩 검증 (유사어 간 cos-similarity, 가장 가까운 token)
    token_pairs_for_valid_embed = [["'m", "am"], ["'s", "is"], ["n't", "not"], ["one", "two"], ["what", "which"],
                                   ["make", "super"], ["deep", "food"], ["what", "could"], ["song", "to"], ["listen", "where"]]
    validate_embedding(A, token_ids, token_pairs_for_valid_embed)

    bert_embedding_dict_df = pd.read_csv('bert_embedding_dict.csv', index_col=0)
    bert_embedding_dict_np = np.array(bert_embedding_dict_df)
    
    tokens_for_valid_embed = ['good', 'better', 'best', 'chicken', 'strawberry', 'blue', 'green', 'down', 'rainy', 'moon']
    for token in tokens_for_valid_embed:
        print_most_similar_tokens(A, token_ids, token_arr, token)

        # BERT 임베딩 결과와 비교
        find_nearest_bert_embedding_rank(token, bert_embedding_dict_np, embed_limit=128)

    # 사용자 입력
    tokenized_input = get_user_input() + (' <null>' * INPUT_TOKEN_CNT_EACH)
    all_outputs = []
    current_turn_outputs = []
    
    next_output = ''
    verbose_for_test = False

    if verbose_for_test:
        print(f'initial tokenized input: {tokenized_input}')

    while True:

        # <person-change> 등장 시, 사용자 입력
        if next_output == '<person-change>':
            ai_output = ' '.join(current_turn_outputs[:-1])
            print(f'AI output :\n{ai_output}')
            current_turn_outputs = []

            # update tokenized input based on user input
            tokenized_input = get_user_input() + (' <null>' * INPUT_TOKEN_CNT_EACH)

        if verbose_for_test:
            print(f'tokenized input : {tokenized_input}')
        
        # 다음 토큰 예측
        next_output_rank = test_model(
            tokenized_input,
            model=mini_chatgpt_model,
            additional_tokenize=False,
            is_return=True,
            token_arr=token_arr,
            token_ids=token_ids,
            verbose=False
        )

        if verbose_for_test:
            for i in range(10):
                print(next_output_rank[i])

        # 재정렬 후 다음 토큰 예측
        restore_next_output_rank(next_output_rank, A, B)

        if verbose_for_test:
            print('')

        if verbose_for_test:
            for i in range(10):
                print(next_output_rank[i])

        # 다음 토큰을 확률적으로 선택
        next_output = choose_one_token(
            next_output_rank,
            verbose=(verbose_for_test and len(all_outputs)==0)
        )
            
        all_outputs.append(next_output)
        current_turn_outputs.append(next_output)

        last_turn = ' '.join(tokenized_input.split(' ')[:INPUT_TOKEN_CNT_EACH])
        current_turn = ' '.join(tokenized_input.split(' ')[INPUT_TOKEN_CNT_EACH + 1:])
            
        tokenized_input = last_turn + ' ' + current_turn + ' ' + next_output

