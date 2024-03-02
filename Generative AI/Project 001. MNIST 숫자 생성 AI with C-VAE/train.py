import pandas as pd
import numpy as np

INPUT_IMG_SIZE = 28
OUTPUT_SIZE = 10


# mnist_train.csv 파일로부터 학습 데이터 추출
def create_train_and_valid_data():
    train_df = pd.read_csv('mnist_train.csv')
    train_n = len(train_df)
    print(train_df)

    train_input = np.zeros((train_n, INPUT_IMG_SIZE, INPUT_IMG_SIZE))
    train_class = np.zeros((train_n, OUTPUT_SIZE))

    for idx, row in train_df.iterrows():
        if idx % 5000 == 0:
            print(idx)

        # input
        inp = np.array(row['1x1':'28x28'].to_list())
        inp = inp.reshape((28, 28))
        inp = inp / 255.0
        train_input[idx] = inp

        # output
        out_class = int(row['label'])
        train_class[idx][out_class] = 1

    return train_input, train_class


if __name__ == '__main__':

    # 학습 데이터 추출 (이미지 input + 해당 이미지의 class)
    train_input, train_class = create_train_and_valid_data()
    print(train_input)
    print(train_class)
