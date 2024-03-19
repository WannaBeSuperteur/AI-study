import tensorflow as tf
from tokenize_data import preprocess_snippet, convert_line_to_tokens, get_input_and_output_for_line
import pandas as pd
import numpy as np

WINDOW_SIZE = 10


# 정성 평가 : 코드를 쭉 읽으면서,empty new line을 추가해야 하는 부분 찾기
def run_test(vocab_map, tokenized_code, original_code, model):
    print('\n\n(nl) means new line')
    tokenized_code_lines = tokenized_code.split(' (nl) ')
    n = len(tokenized_code_lines)
    print(f'lines: {n}\n')

    print(vocab_map)

    # tokenized code를 line 단위로 split 하기
    tokenized_lines = [x.strip() for x in tokenized_code_lines]

    converted_code = original_code[0] + original_code[1]

    print(' ==== original code ====')
    for i in range(n):
        print('%5d |' % (i + 1), original_code[i].split('\n')[0])

    print('\n ==== test ====')
    for i in range(n - 3):

        # 코드를 line 단위로 모델의 입력 데이터 token으로 변환 처리
        # vocab map을 이용하여 token ID로 변환
        model_input = get_input_and_output_for_line(tokenized_lines, i, n)[0]
        model_input = [vocab_map[vocab_map['token'] == x].iloc[0]['token_id'] for x in model_input]
        model_input = np.array([model_input])

        # 메인 모델을 통한 테스트 및 결과 출력
        test_result = main_model(model_input)[0][0]

        if test_result >= 0.5:
            print(f'Empty new line should be between line {i + 2} and line {i + 3} with prob {round(float(test_result), 6)}.')
            converted_code += '\n' + original_code[i + 2]
        else:
            print(f'Empty new line between line {i + 2} and line {i + 3} with prob {round(float(test_result), 6)}.')
            converted_code += original_code[i + 2]

    converted_code += original_code[-1]

    print('\n ==== AI converted code ====\n')
    print(converted_code)


# 정량 평가 결과의 metric 값 출력
def print_metrics(test_output, ground_truth):
    test_output_flatten = np.array(test_output).flatten()
    ground_truth_flatten = np.array(ground_truth).flatten()
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
    print(f'recall    : {recall}')
    print(f'precision : {precision}')
    print(f'F1 score  : {f1_score}')
    

# 정량 평가 실시
def run_quantitative_test(model):

    # 테스트 데이터 읽기
    test_data = pd.read_csv('train_data_token_id_for_test.csv', index_col=0)
    print(f'test data:\n{test_data}')

    test_input = np.array(test_data)[:, :-1].astype(np.int64)
    ground_truth = np.array(test_data)[:, -1:]
    
    # vocab map 가져오기
    vocab_map = pd.read_csv('vocab_map.csv', index_col=0)

    # main model을 이용하여 테스트
    test_output = model(test_input)

    # accuracy, recall, precision, F1 score 출력
    print_metrics(test_output, ground_truth)


# 정성 평가 실시
def run_qualitative_test(model):
    f = open('python_code.txt', 'r')
    lines = f.readlines()
    f.close()

    # vocab map 가져오기
    vocab_map = pd.read_csv('vocab_map.csv', index_col=0)

    # 코드 토큰화
    tokenized_code = preprocess_snippet(''.join(lines))

    # main model을 이용하여 테스트
    run_test(vocab_map, tokenized_code, original_code=lines, model=model)


if __name__ == '__main__':

    # 모델 로딩
    main_model = tf.keras.models.load_model('main_model')

    # 정량 평가 실시
    run_quantitative_test(main_model)

    # 정성 평가 실시
    run_qualitative_test(main_model)
