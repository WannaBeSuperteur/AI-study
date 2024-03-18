import tensorflow as tf
from tokenize_data import preprocess_snippet, convert_line_to_tokens
import pandas as pd
import numpy as np

WINDOW_SIZE = 10


# 코드를 tokenize 처리 (각 line 별로)
def tokenize_lines(lines):
    preprocessed_snippets = preprocess_snippet(''.join(lines))
    pass


# 정성 평가 : 코드를 쭉 읽으면서,empty new line을 추가해야 하는 부분 찾기
def run_test(vocab_map, tokenized_lines, main_model):
    print('\n\n(nl) means new line\n')
    n = len(tokenized_lines)

    for i in range(n - 3):

        # 코드를 line 단위로 모델의 입력 데이터 token으로 변환 처리
        input_1 = convert_line_to_tokens(before_line_1.split(' '), direction='end')
        input_2 = convert_line_to_tokens(before_line_2.split(' '), direction='end')
        input_3 = convert_line_to_tokens(after_line_1.split(' '), direction='start')
        input_4 = convert_line_to_tokens(after_line_2.split(' '), direction='start')

        input_merged = input_1 + input_2 + input_3 + input_4

        # 메인 모델을 통한 테스트 및 결과 출력
        test_result = main_model([input_merged])[0][0]

        if test_result >= 0.5:
            print(f'Empty new line should be between "{input_2}" and "{input_3}" with prob {round(float(test_result), 6)}.')


# 정량 평가 결과의 metric 값 출력
def print_metrics(test_output, ground_truth):
    test_output_flatten = test_output.flatten()
    ground_truth_flatten = ground_truth.flatten()
    n = len(test_output_flatten)

    TP = 0 # True Positive
    TN = 0 # True Negative
    FP = 0 # False Positive
    FN = 0 # False Negative

    for i in range(n):
        output = test_output_flatten[i]
        gt = ground_truth_flatten[i]
        
        if output >= 0.5 and gt == 1.0:
            TP += 1

        elif output < 0.5 and gt == 1.0:
            FN += 1

        elif output >= 0.5 and gt == 0.0:
            FP += 1

        elif output < 0.5 and gt == 0.0:
            TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * recall * precision / (recall + precision)

    print(f'True  Positive : {TP}')
    print(f'True  Negative : {TN}')
    print(f'False Positive : {FP}')
    print(f'False Negative : {FN}')

    print(f'\naccuracy  : {accuracy}')
    print(f'\nrecall    : {recall}')
    print(f'\nprecision : {nprecision}')
    print(f'\nF1 score  : {f1_score}')
    

# 정량 평가 실시
def run_quantitative_test():

    # 테스트 데이터 읽기
    test_data = pd.read_csv('train_data_token_id_for_test.csv', index_col=0)
    print(f'test data:\n{test_data}')

    test_input = np.array(test_data)[:, :-1]
    ground_truth = np.array(test_data)[:, -1:]
    
    # vocab map 가져오기
    vocab_map = pd.read_csv('vocab_map.csv', index_col=0)

    # 코드 토큰화
    tokenized_lines = tokenize_lines(lines, vocab_map)

    # main model을 이용하여 테스트
    main_model = tf.keras.models.load_model('main_model')
    test_output = main_model(test_input)

    # accuracy, recall, precision, F1 score 출력
    print_metrics(test_output, ground_truth)


# 정성 평가 실시
def run_qualitative_test():
    f = open('python_code.txt', 'r')
    lines = convert_code(f.readlines())
    f.close()

    print(lines)

    # vocab map 가져오기
    vocab_map = pd.read_csv('vocab_map.csv', index_col=0)

    # 코드 토큰화
    tokenized_lines = tokenize_lines(lines, vocab_map)

    # main model을 이용하여 테스트
    main_model = tf.keras.models.load_model('main_model')
    run_test(vocab_map, tokenized_lines, main_model)


if __name__ == '__main__':

    # 정량 평가 실시
    run_quantitative_test()

    # 정성 평가 실시
    run_qualitative_test()
