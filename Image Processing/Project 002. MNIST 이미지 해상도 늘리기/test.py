import tensorflow as tf
from PIL import Image

import pandas as pd
import numpy as np
import os


IMAGE_SIZE = 28
model = tf.keras.models.load_model('main_model')


def create_img_dir():
    if 'test_outputs' not in os.listdir():
        os.makedirs('test_outputs')


# 테스트 데이터 읽기
def read_test_data():
    test_df = pd.read_csv('mnist_test.csv', index_col=0)
    return test_df


# 테스트 데이터 배열로부터 실제 테스트 데이터 추출
def extract_test_data_each_img(test_np):
    test_data_for_model = []
    test_np_2d = test_np.reshape((IMAGE_SIZE, IMAGE_SIZE))
    test_pad = IMAGE_SIZE // 4 - 1

    # (28 x 28) -> ((6 + 28 + 6) x (6 + 28 + 6)) = (40, 40)
    test_np_2d_pad = np.pad(test_np_2d, ((test_pad, test_pad), (test_pad, test_pad)), 'constant', constant_values=0)

    for i in range(test_pad, test_pad + IMAGE_SIZE - 1):
        for j in range(test_pad, test_pad + IMAGE_SIZE - 1):
            top = i - test_pad
            bottom = i + test_pad + 2
            left = j - test_pad
            right = j + test_pad + 2

            test_data_for_model.append(test_np_2d_pad[top:bottom, left:right].flatten())

    return test_data_for_model


# 최초 (limit) 장의 이미지에서 테스트 데이터 추출
def extract_test_data(test_df, limit=100):
    entire_test_data_for_model = []

    for i in range(limit):
        test_np = np.array(test_df.iloc[i])
        test_data_for_model = extract_test_data_each_img(test_np)
        entire_test_data_for_model.append(test_data_for_model)

    return np.array(entire_test_data_for_model)


# 개별 이미지에 대해 테스트 이미지 생성
# model_test_data_for_img : with shape ((27 * 27, 14 * 14)) = ((729, 196))
def generate_test_data_for_img(model_test_data_for_img):
    NEW_IMG_SIZE = 2 * IMAGE_SIZE - 1

    # central pixel 4개 중 왼쪽 위, 오른쪽 위, 왼쪽 아래, 오른쪽 아래 픽셀의 인덱스
    CENTER_TOP_LEFT = (IMAGE_SIZE // 2) * (IMAGE_SIZE // 4 - 1) + (IMAGE_SIZE // 4 - 1)
    CENTER_TOP_RIGHT = (IMAGE_SIZE // 2) * (IMAGE_SIZE // 4 - 1) + (IMAGE_SIZE // 4)
    CENTER_BOTTOM_LEFT = (IMAGE_SIZE // 2) * (IMAGE_SIZE // 4) + (IMAGE_SIZE // 4 - 1)
    CENTER_BOTTOM_RIGHT = (IMAGE_SIZE // 2) * (IMAGE_SIZE // 4) + (IMAGE_SIZE // 4)
    
    result_img = np.zeros((NEW_IMG_SIZE, NEW_IMG_SIZE))

    # fill central pixels
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            if i < IMAGE_SIZE - 1 and j < IMAGE_SIZE - 1:
                result_img[2 * i][2 * j] = model_test_data_for_img[(IMAGE_SIZE - 1) * i + j][CENTER_TOP_LEFT]
                
            elif i == IMAGE_SIZE - 1 and j < IMAGE_SIZE - 1:
                result_img[2 * i][2 * j] = model_test_data_for_img[(IMAGE_SIZE - 1) * (i-1) + j][CENTER_BOTTOM_LEFT]                

            elif i < IMAGE_SIZE - 1 and j == IMAGE_SIZE - 1:
                result_img[2 * i][2 * j] = model_test_data_for_img[(IMAGE_SIZE - 1) * i + (j-1)][CENTER_TOP_RIGHT]

            elif i == IMAGE_SIZE - 1 and j == IMAGE_SIZE - 1:
                result_img[2 * i][2 * j] = model_test_data_for_img[(IMAGE_SIZE - 1) * (i-1) + (j-1)][CENTER_BOTTOM_RIGHT]
                
    # fill other pixels
    model_output_sum = np.zeros((NEW_IMG_SIZE, NEW_IMG_SIZE))
    model_output_cnt = np.zeros((NEW_IMG_SIZE, NEW_IMG_SIZE))
    
    for i in range(IMAGE_SIZE - 1):
        for j in range(IMAGE_SIZE - 1):
            input_cnn = model_test_data_for_img[i * (IMAGE_SIZE - 1) + j]
            input_cnn = np.reshape(input_cnn, (-1, len(input_cnn)))
            
            input_center_tl = input_cnn[0][CENTER_TOP_LEFT]
            input_center_tr = input_cnn[0][CENTER_TOP_RIGHT]
            input_center_bl = input_cnn[0][CENTER_BOTTOM_LEFT]
            input_center_br = input_cnn[0][CENTER_BOTTOM_RIGHT]

            input_center = np.array([[input_center_tl, input_center_tr, input_center_bl, input_center_br]])

            # input to model
            test_input = np.concatenate([input_cnn, input_center], axis=1)
            test_output = np.array(model(test_input))

            # position of outputs A, B, C, D and E, respectively (shape of '+')
            cross_axis_add = [0, 1], [1, 0], [1, 1], [1, 2], [2, 1]
            
            for idx, axis_add in enumerate(cross_axis_add):
                model_output_sum[2 * i + axis_add[0]][2 * j + axis_add[1]] = float(test_output[0][idx])
                model_output_cnt[2 * i + axis_add[0]][2 * j + axis_add[1]] += 1

    # final output image
    for i in range(NEW_IMG_SIZE):
        for j in range(NEW_IMG_SIZE):
            if model_output_cnt[i][j] > 0:
                result_img[i][j] = model_output_sum[i][j] / model_output_cnt[i][j]

    return result_img
            

# 테스트 결과 이미지 생성 및 저장
# entire_test_data_for_model : with shape ((N, 27 * 27, 14 * 14)) = ((N, 729, 196))
def generate_test_result(entire_test_data_for_model):
    model_test_data = entire_test_data_for_model / 255.0
    
    for i in range(len(model_test_data)):
        print(f'generating image {i} ...')
        
        img = generate_test_data_for_img(model_test_data[i])

        # 이미지 저장
        img_np = img * 255.0
        img_np_rgb = Image.fromarray(img_np).convert('RGB')
        img_np_rgb.save(f'test_outputs/test_output_{i}.png')
        

if __name__ == '__main__':
    create_img_dir()    

    # 테스트 데이터 수집
    test_df = read_test_data()
    print(test_df)

    # 모델에 넣을 테스트 데이터로 변환
    entire_test_data_for_model = extract_test_data(test_df)

    print(np.shape(entire_test_data_for_model))
    print(np.array(entire_test_data_for_model))

    # 테스트 결과 이미지 생성 및 저장
    generate_test_result(entire_test_data_for_model)
