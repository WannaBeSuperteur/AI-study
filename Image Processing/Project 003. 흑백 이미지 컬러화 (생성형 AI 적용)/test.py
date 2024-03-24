import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
import math

from test_helper import create_hsv_image

RESIZE_HEIGHT = 144
RESIZE_WIDTH = 112
CROP_HEIGHT = (RESIZE_HEIGHT - RESIZE_WIDTH) // 2

HIDDEN_DIMS = 200
BATCH_SIZE = 32
NUM_CLASSES = 16

decoder_model = tf.keras.models.load_model('main_vae_decoder')


def crop_and_resize_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#    resized_image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))
#    return resized_image[CROP_HEIGHT:-CROP_HEIGHT, :]
    resized_image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_WIDTH))
    return resized_image


# test_images 폴더에 있는 모든 사진을 crop -> resize 하여 저장
def crop_and_resize():
    test_imgs = os.listdir('test_images')
    for img_name in test_imgs:
        modified_img = crop_and_resize_img(f'test_images/{img_name}')
        cv2.imwrite(f'test_images_resized/{img_name}', modified_img)


# test_images_resized 및 test_output 디렉토리 생성
def create_test_resized_and_output_dir():
    try:
        os.makedirs('test_images_resized')
    except:
        pass
    
    try:
        os.makedirs('test_output')
    except:
        pass


# 최종 테스트 이미지 생성
def generate_test_result_image(image, coord_x, coord_y, img_name):
    print('\n ==== coord_x ====')
    print(coord_x)

    print('\n ==== coord_y ====')
    print(coord_y)

    hsv_array = create_hsv_image(image, coord_x, coord_y, img_size=RESIZE_WIDTH)

    print('\n ==== HSV ====')
    print(hsv_array)
    
    final_image = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)

    print('\n ==== final image ====')
    print(final_image)
    
    cv2.imwrite(f'test_output/{img_name}.png', final_image)


# 개별 이미지에 대한 테스트 실시
def run_test_for_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.reshape(image, (1, RESIZE_WIDTH, RESIZE_WIDTH))
    image = image / 255.0
    
    image_class = np.zeros((1, NUM_CLASSES))
    image_class_no = int(path.split('/')[-1].split('_')[1])
    image_class[0][image_class_no] = 1.0

    for i in range(5):
        latent_space = np.random.normal(0.0, 1.0, size=(1, HIDDEN_DIMS))

        coord_x_and_y = decoder_model([latent_space, image, image_class])
        coord_x_and_y = np.array(coord_x_and_y)
        coord_x = coord_x_and_y[0, :, :, 0]
        coord_y = coord_x_and_y[0, :, :, 1]

        # 최종 테스트 이미지 만들기
        generate_test_result_image(image, coord_x, coord_y, img_name=f'{path.split("/")[-1]}_{i}')


# 테스트 실시
def run_test():
    test_imgs = os.listdir('test_images_resized')
    for img_name in test_imgs:
        run_test_for_img(f'test_images_resized/{img_name}')


if __name__ == '__main__':
    np.set_printoptions(linewidth=250)

    create_test_resized_and_output_dir()
    crop_and_resize()
    run_test()
    
