import pandas as pd
import numpy as np


# 입력 및 출력 데이터의 row 추출

# i: 0이면 짝수 번째 row, 1이면 홀수 번째 row 를 입력 데이터로
# j: 0이면 짝수 번째 column, 1이면 홀수 번째 column을 입력 데이터로
# 출력 데이터는 입력 데이터의 정중앙 4개 pixel이 이루는 3 x 3 정사각형의 해당 4개 pixel을 제외한 나머지 5개 pixel 값

def create_input_and_output_row(original_row, i, j, input_data, output_data):
    input_row = []
    output_row = []

    # input 14 x 14 = 196 cells
    for k in range(14):
        for l in range(14):
            input_row.append(original_row[28 * (2 * k + i) + (2 * l + j)])

    # output 5 cells
    output_row.append(original_row[28 * (12 + i) + 13 + j])
    output_row.append(original_row[28 * (13 + i) + 12 + j])
    output_row.append(original_row[28 * (13 + i) + 13 + j])
    output_row.append(original_row[28 * (13 + i) + 14 + j])
    output_row.append(original_row[28 * (14 + i) + 13 + j])

    input_data.append(input_row)
    output_data.append(output_row)


# 입력 데이터 및 출력 데이터 추출
def get_input_and_output(data):
    input_data = []
    output_data = []
    row_cnt = 0

    for idx, row in data.iterrows():
        if row_cnt % 5000 == 0:
            print(row_cnt)
            
        row_ = list(row)
        for i in range(2):
            for j in range(2):
                create_input_and_output_row(row_, i, j, input_data, output_data)

        row_cnt += 1

    return np.array(input_data), np.array(output_data)


# 원본 데이터 mnist_test.csv, mnist_train.csv 로부터 본 프로젝트용 데이터 생성
def create_data(data, save_name):
    input_data, output_data = get_input_and_output(data)

    print(f'input_data ({save_name}) : {np.shape(input_data)}\n{input_data}')
    print(f'output_data ({save_name}) : {np.shape(output_data)}\n{output_data}')
    
    data = np.concatenate([input_data, output_data], axis=1)
    data = pd.DataFrame(data)
    data.columns = [f'input_{x}' for x in range(14 * 14)] + [f'output_{x}' for x in range(5)]
    
    data.to_csv(f'{save_name}.csv')


if __name__ == '__main__':
    train_data = pd.read_csv('mnist_train.csv', index_col=0)
    test_data = pd.read_csv('mnist_test.csv', index_col=0)

    create_data(train_data, 'train_data')
    create_data(test_data, 'test_data')
    
