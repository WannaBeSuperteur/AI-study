# settings for MODEL 28

import tensorflow as tf
import numpy as np
from PIL import Image
import os

HIDDEN_DIMS = 231
RANDOM_LATENT_COUNT = 15
BATCH_SIZE = 32

info_settings = [
    {'setting_no': 0, 'male_prob': 0.0000001, 'female_prob': 0.9999999, 'hair_color': 0.05, 'mouth': 1.0, 'eyes': 1.0},
    {'setting_no': 1, 'male_prob': 0.0000001, 'female_prob': 0.9999999, 'hair_color': 0.5,  'mouth': 1.0, 'eyes': 1.0},
    {'setting_no': 2, 'male_prob': 0.0000001, 'female_prob': 0.9999999, 'hair_color': 0.75, 'mouth': 1.0, 'eyes': 1.0},
    {'setting_no': 3, 'male_prob': 0.0000001, 'female_prob': 0.9999999, 'hair_color': 1.0,  'mouth': 1.0, 'eyes': 1.0},
    {'setting_no': 5, 'male_prob': 0.0000001, 'female_prob': 0.9999999, 'hair_color': 1.0,  'mouth': 0.0, 'eyes': 1.0},
    {'setting_no': 6, 'male_prob': 0.0000001, 'female_prob': 0.9999999, 'hair_color': 1.0,  'mouth': 0.0, 'eyes': 0.5},
    {'setting_no': 7, 'male_prob': 0.9999999, 'female_prob': 0.0000001, 'hair_color': 1.0,  'mouth': 0.0, 'eyes': 1.0}
]


# RGB <-> BGR
def convert_RGB_to_BGR(original_image):
    return original_image[:, :, ::-1]


# 생성된 cvae_decoder_model 테스트
def test_decoder(cvae_decoder, info_setting, latent_vector, latent_vector_no):
    male_prob = info_setting['male_prob']
    female_prob = info_setting['female_prob']
    hair_color = info_setting['hair_color']
    mouth = info_setting['mouth']
    eyes = info_setting['eyes']

    input_info_one = [male_prob, female_prob, hair_color, 1.0 - hair_color, mouth, eyes,
                      np.random.uniform(),  # face location from top
                      np.random.uniform(),  # face location from left
                      np.random.uniform(),  # face location from right
                      0.975,  # background mean
                      0.2]  # background std

    input_info = np.array([input_info_one for _ in range(BATCH_SIZE)])

    img = cvae_decoder([latent_vector, input_info])
    img_np = np.array(img.numpy() * 255.0, dtype=np.uint8)
    img_np_rgb = Image.fromarray(convert_RGB_to_BGR(img_np[0]))

    setting_no = info_setting['setting_no']
    img_np_rgb.save(f'test_outputs_info_change/test_output_{setting_no}_{latent_vector_no}.png')


# 모든 케이스에 대해 decoder model 테스트
def test_all_cases(cvae_decoder):
    random_latent_cnt = RANDOM_LATENT_COUNT

    latent_vectors = []
    for _ in range(random_latent_cnt):
        latent_vector = np.random.normal(0.0, 1.0, size=(BATCH_SIZE, HIDDEN_DIMS))
        latent_vectors.append(latent_vector)

    for info_setting in info_settings:
        for latent_vector_no in range(random_latent_cnt):
            test_decoder(cvae_decoder, info_setting, latent_vectors[latent_vector_no], latent_vector_no)


# test_outputs 디렉토리 생성
def create_test_outputs_dir():
    try:
        os.makedirs('test_outputs_info_change')
    except:
        pass


if __name__ == '__main__':
    cvae_decoder = tf.keras.models.load_model('cvae_decoder_model')
    create_test_outputs_dir()

    test_all_cases(cvae_decoder)
