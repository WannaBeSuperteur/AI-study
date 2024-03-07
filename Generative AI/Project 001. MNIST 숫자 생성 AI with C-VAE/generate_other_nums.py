import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import os

from train import noise_maker

BATCH_SIZE = 32
HIDDEN_DIMS = 40
IMAGE_SIZE = 28

# MUST BE AFTER BATCH_SIZE, HIDDEN_DIMS, IMAGE_SIZE INIT !!
from test import generate_image_with_num


classify_model = tf.keras.models.load_model('classify_nums_model')
encoder_model = tf.keras.models.load_model('cvae_encoder_model', custom_objects={'noise_maker': noise_maker})
decoder_model = tf.keras.models.load_model('cvae_decoder_model')


# 숫자 이미지 읽기
def read_number_img():
    img = cv2.imread('number.png', cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    return img


# 어떤 숫자인지 판단
def get_class(num_img):
    num_img_ = num_img.reshape((-1, IMAGE_SIZE, IMAGE_SIZE))
    num_class_probs = classify_model(num_img_)
    num_class = np.argmax(num_class_probs[0])
    
    print(f'probs :\n{np.array(num_class_probs[0])}')
    print(f'argmax :\n{num_class}')

    return num_class


# Encoder 에 넣어서 latent vector 계산
def get_latent_vector(img, num_class, white_ratio):
    img_ = img.reshape((-1, IMAGE_SIZE, IMAGE_SIZE))
    conditions = np.zeros((1, 11))
    conditions[0][num_class] = 1
    conditions[0][10] = white_ratio
    
    latent_vector = encoder_model([img_, conditions])
    return latent_vector


# white_ratio 0.0, 0.5, 1.0 으로 나머지 class (예: 2 -> 0, 1, 3, ..., 9) 이미지 생성
def generate_img(num_class):
    for i in range(10):
        if i == num_class:
            continue
        
        for bold_idx in range(3):
            latent_vector = get_latent_vector(
                img=num_img,
                num_class=i,
                white_ratio=(bold_idx / 2.0)
            )

            for latent_vector_no in range(8):
                generate_image_with_num(
                    num=i,
                    bold=(bold_idx / 2.0),
                    latent_space=latent_vector[latent_vector_no:latent_vector_no+1], # n-th (n=1,2,3,4) latent vector
                    prefix=f'generated_{latent_vector_no}_{bold_idx}'
                )


if __name__ == '__main__':
    if 'test_imgs' not in os.listdir():
        os.makedirs('test_imgs')
    
    num_img = read_number_img()
    num_class = get_class(num_img)

    # 나머지 class 의 이미지 생성
    generate_img(num_class)

