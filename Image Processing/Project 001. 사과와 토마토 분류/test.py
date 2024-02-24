import pandas as pd
import numpy as np
import math

import tensorflow as tf


# accuracy 측정
def compute_accuracy(arr_output, arr_gt):
    arr_output_np = np.array(arr_output)
    arr_gt_np = np.array(arr_gt)
    
    return np.sum(arr_output_np == arr_gt_np) / len(arr_gt)


# 특정 class에 대한 precision 측정
def compute_precision(arr_output, arr_gt, class_str):
    n = len(arr_gt)
    TP = 0
    TP_FP = 0

    for i in range(n):
        if arr_output[i] == class_str:
            TP_FP += 1

            if arr_gt[i] == class_str:
                TP += 1

    return TP / TP_FP


# 특정 class에 대한 recall 측정
def compute_recall(arr_output, arr_gt, class_str):
    n = len(arr_gt)
    TP = 0
    TP_FN = 0

    for i in range(n):
        if arr_gt[i] == class_str:
            TP_FN += 1

            if arr_output[i] == class_str:
                TP += 1

    return TP / TP_FN


# [a, b] 에서 a가 더 크면 사과, b가 더 크면 토마토로 변환
def convert_arr_to_result(arr):
    if arr[0] >= arr[1]:
        return 'apple'
    else:
        return 'tomato'


# F1 score
def compute_F1(precision, recall):
    return 2 * precision * recall / (precision + recall)


# 모델 테스트 결과에 의한 성능 측정
def print_performance_scores(test_output, ground_truth):

    # map을 이용하여 str array로 변환
    test_output = list(map(convert_arr_to_result, test_output))
    ground_truth = list(map(convert_arr_to_result, ground_truth))

    # 성능지표 측정 및 출력
    accuracy = compute_accuracy(test_output, ground_truth)
    print('accuracy:', accuracy)

    for c in ['apple', 'tomato']:
        c_prec = compute_precision(test_output, ground_truth, c)
        c_recl = compute_recall(test_output, ground_truth, c)
        c_F1 = compute_F1(c_prec, c_recl)

        print(f'precision for {c} : {c_prec}')
        print(f'recall    for {c} : {c_recl}')
        print(f'F1 score  for {c} : {c_F1}')


if __name__ == '__main__':

    # [apple, apple, apple, apple, apple, tomato, tomato, tomato]
    test_prediction = [[1, 0], [1, 0], [0.9, 0.1], [0.75, 0.25], [0.9, 0.1],
                       [0.3, 0.7], [0.45, 0.55], [0.2, 0.8]]

    # [apple, apple, tomato, tomato, apple, tomato, apple, tomato]
    ground_truth = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0],
                    [0, 1], [1, 0], [0, 1]]

    print_performance_scores(test_prediction, ground_truth)
