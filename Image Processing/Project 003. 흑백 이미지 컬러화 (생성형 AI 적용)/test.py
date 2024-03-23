import pandas as pd
import numpy as np
import cv2
import os

RESIZE_HEIGHT = 144
RESIZE_WIDTH = 112
CROP_HEIGHT = (RESIZE_HEIGHT - RESIZE_WIDTH) // 2


def crop_and_resize_img(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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


if __name__ == '__main__':
    crop_and_resize()
    create_test_output_dir()
    
