import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
import math

RESIZE_HEIGHT = 144
RESIZE_WIDTH = 112
CROP_HEIGHT = (RESIZE_HEIGHT - RESIZE_WIDTH) // 2

HIDDEN_DIMS = 200
BATCH_SIZE = 32

decoder_model = tf.keras.models.load_model('main_vae_decoder')


def crop_and_resize_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (RESIZE_WIDTH, RESIZE_HEIGHT))
    return resized_image[CROP_HEIGHT:-CROP_HEIGHT, :]


# test_images 폴더에 있는 모든 사진을 crop -> resize 하여 저장
def crop_and_resize():
    test_imgs = os.listdir('test_images')
    for img_name in test_imgs:
        modified_img = crop_and_resize_img(f'test_images/{img_name}')
        cv2.imwrite(f'test_images_resized/{img_name}', modified_img)


# test_output 디렉토리 생성
def create_test_output_dir():
    try:
        os.makedirs('test_output')
    except:
        pass


# compute hue
def compute_hue(x, y):
    if x == 0:
        if y > 0:
            return 45.0
        else:
            return 135.0

    # 제1, 2 사분면
    elif y >= 0:
        return np.arctan2(y, x) * (180.0 / math.pi) / 2.0

    # 제3, 4 사분면
    else:
        return (np.arctan2(y, x) + 2.0 * math.pi) * (180.0 / math.pi) / 2.0


# 최종 테스트 이미지 생성
def generate_test_result_image(image, coord_x, coord_y, img_name):
    hsv_array = np.zeros((RESIZE_WIDTH, RESIZE_WIDTH, 3))

    print('\n ==== coord_x ====')
    print(coord_x)

    print('\n ==== coord_y ====')
    print(coord_y)

    for i in range(RESIZE_WIDTH):
        for j in range(RESIZE_WIDTH):
            hue = compute_hue(coord_x[i][j], coord_y[i][j])
            saturation = math.sqrt(coord_x[i][j] * coord_x[i][j] + coord_y[i][j] * coord_y[i][j])
            brightness = image[0][i][j]

            hsv_array[i][j][0] = hue
            hsv_array[i][j][1] = saturation
            hsv_array[i][j][2] = 255.0 * brightness

    hsv_array = np.array(hsv_array).astype(np.float32)

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

    for i in range(5):
        latent_space = np.random.normal(0.0, 1.0, size=(1, HIDDEN_DIMS))

        coord_x_and_y = decoder_model([latent_space, image])
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
    np.set_printoptions(linewidth=160)
    
    crop_and_resize()
    create_test_output_dir()
    run_test()
    
