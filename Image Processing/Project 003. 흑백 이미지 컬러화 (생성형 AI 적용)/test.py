import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
import math

from test_helper import create_hsv_image, create_lv_1_2_3_images

RESIZE_HEIGHT = 144
RESIZE_WIDTH = 112
CROP_HEIGHT = (RESIZE_HEIGHT - RESIZE_WIDTH) // 2
COLORIZE_MAP_SIZE = RESIZE_WIDTH // 8

HIDDEN_DIMS = 120
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


# hue, brightness 와 saturation = max(R, G, B) - min(R, G, B) 을 이용하여 이미지 복원
def restore_final_image(hsv_array, img_size):
    final_image = np.zeros((img_size, img_size, 3))
    
    for i in range(img_size):
        for j in range(img_size):
            hue = hsv_array[i][j][0]
            saturation = hsv_array[i][j][1] * 255.0
            brightness = hsv_array[i][j][2] # brightness = value

            rgb_max = brightness + saturation / 2.0
            rgb_min = brightness - saturation / 2.0
            rgb_mid_ratio = abs((hue % 120.0) - 60.0) / 60.0
            rgb_mid = rgb_min + (rgb_max - rgb_min) * rgb_mid_ratio

            if hue < 60:
                R = rgb_max
                G = rgb_mid
                B = rgb_min

            elif hue < 120:
                R = rgb_mid
                G = rgb_max
                B = rgb_min

            elif hue < 180:
                R = rgb_min
                G = rgb_max
                B = rgb_mid

            elif hue < 240:
                R = rgb_min
                G = rgb_mid
                B = rgb_max

            elif hue < 300:
                R = rgb_mid
                G = rgb_min
                B = rgb_max

            else:
                R = rgb_max
                G = rgb_min
                B = rgb_mid

            final_image[i][j][0] = B
            final_image[i][j][1] = G
            final_image[i][j][2] = R

    return final_image


# add gaussian blur
def add_gaussian_blur(image_2d):
    return cv2.GaussianBlur(image_2d, ksize=(0, 0), sigmaX=3, sigmaY=3)


# 최종 테스트 이미지 생성
def generate_test_result_image(image, coord_x, coord_y, img_name):
    print(f'\nimage name : {img_name}')
    
    print('\n ==== coord_x ====')
    print(coord_x)

    print('\n ==== coord_y ====')
    print(coord_y)

    hsv_array = create_hsv_image(image, coord_x, coord_y, img_size=RESIZE_WIDTH)

    print('\n ==== HSV ====')
    print(hsv_array)

    # add gaussian blur
    hsv_array[:, :, 0] = add_gaussian_blur(hsv_array[:, :, 0]) # hue
    hsv_array[:, :, 1] = add_gaussian_blur(hsv_array[:, :, 1]) # saturation

    print('\n ==== HSV with Gaussian blur ====')
    print(hsv_array)
    
#    final_image = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)
    final_image = restore_final_image(hsv_array, img_size=RESIZE_WIDTH)

    print('\n ==== final image ====')
    print(final_image)
    
    cv2.imwrite(f'test_output/{img_name}.png', final_image)


# 개별 이미지에 대한 테스트 실시
def run_test_for_img(path):
    print(f'run test for image {path} ...')
    
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.reshape(image, (1, RESIZE_WIDTH, RESIZE_WIDTH))
    image = image / 255.0
    
    image_resized = cv2.resize(image[0], (COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE))
    image_resized = np.array([image_resized])
    
    for t in range(5):
        latent_space = np.random.normal(0.0, 1.0, size=(1, HIDDEN_DIMS))
        coord_x_and_y = np.zeros((1, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 2))

        # find x and y coord values for each 8 x 8 area
        for i in range(COLORIZE_MAP_SIZE):
            for j in range(COLORIZE_MAP_SIZE):

                # generate lv1 ~ lv3 image
                lv1_image, lv2_image, lv3_image = create_lv_1_2_3_images(
                    image_size=RESIZE_WIDTH,
                    color_map_size=COLORIZE_MAP_SIZE,
                    greyscale_image=image[0],
                    i_start=i * 8,
                    j_start=j * 8
                )

                # reshape lv1 ~ lv3 image for model
                lv1_image = np.reshape(lv1_image, (-1, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE))
                lv2_image = np.reshape(lv2_image, (-1, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE))
                lv3_image = np.reshape(lv3_image, (-1, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE))

                if t == 0 and i == 0 and j == 0:
                    print(f'shape of latent space  : {np.shape(latent_space)}')
                    print(f'shape of image_resized : {np.shape(image_resized)}')
                    print(f'shape of lv1_image     : {np.shape(lv1_image)}')
                    print(f'shape of lv2_image     : {np.shape(lv2_image)}')
                    print(f'shape of lv3_image     : {np.shape(lv3_image)}')

                # decoder output -> x, y coord value
                coord_x_y = decoder_model([
                    latent_space, image_resized, lv1_image, lv2_image, lv3_image
                ])

                if t == 0 and i == 0 and j == 0:
                    print(coord_x_y)
                    print(f'shape of coord_x_and_y : {np.shape(coord_x_and_y)}')
                    print(f'shape of coord_x_y     : {np.shape(coord_x_y)}')
                
                coord_x_and_y[0][i][j][0] = coord_x_y[0][0] # x coord value
                coord_x_and_y[0][i][j][1] = coord_x_y[0][1] # y coord value

        coord_x = coord_x_and_y[0, :, :, 0]
        coord_y = coord_x_and_y[0, :, :, 1]

        # 최종 테스트 이미지 만들기
        generate_test_result_image(image, coord_x, coord_y, img_name=f'{path.split("/")[-1]}_{t}')


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
    
