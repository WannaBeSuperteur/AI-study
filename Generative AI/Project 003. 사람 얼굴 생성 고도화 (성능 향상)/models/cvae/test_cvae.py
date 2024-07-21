# from GAI-P2


import tensorflow as tf
import numpy as np
import os
from PIL import Image

HIDDEN_DIMS = 120  # same as model setting
RANDOM_LATENT_COUNT = 15
BATCH_SIZE = 32

BACKGROUND_FIXED = 0.99

info_settings = [
    {'setting_no': 0, 'hair_color': 0.05, 'mouth': 1.0, 'eyes': 1.0, 'head': 0.5},
    {'setting_no': 1, 'hair_color': 0.5,  'mouth': 1.0, 'eyes': 1.0, 'head': 0.5},
    {'setting_no': 2, 'hair_color': 0.75, 'mouth': 1.0, 'eyes': 1.0, 'head': 0.5},
    {'setting_no': 3, 'hair_color': 1.0,  'mouth': 1.0, 'eyes': 1.0, 'head': 0.5},
    {'setting_no': 4, 'hair_color': 1.0,  'mouth': 0.0, 'eyes': 1.0, 'head': 0.5},  # 입을 다묾
    {'setting_no': 5, 'hair_color': 1.0,  'mouth': 1.0, 'eyes': 0.5, 'head': 0.5},  # 눈을 반쯤 감음
    {'setting_no': 6, 'hair_color': 1.0,  'mouth': 1.0, 'eyes': 1.0, 'head': 0.0},  # 고개를 왼쪽으로 돌림
    {'setting_no': 7, 'hair_color': 1.0,  'mouth': 1.0, 'eyes': 1.0, 'head': 1.0}   # 고개를 오른쪽으로 돌림
]


# RGB <-> BGR
def convert_RGB_to_BGR(original_image):
    return original_image[:, :, ::-1]


def test_decoder(cvae_decoder, info_setting, latent_vector, latent_vector_no, epoch_no):
    """
    test CVAE decoder with a specific case from 'info_settings'

    Args:
        cvae_decoder     (TensorFlow Model) : CVAE decoder
        info_setting     (dict)             : specific case to test
        latent_vector                       : latent vector for decoder input
        latent_vector_no (int)              : latent vector No. (0, 1, ..., RANDOM_LATENT_COUNT - 1)
        epoch_no         (int or str)       : epoch number ('finished' for already trained model)
    """

    hair_color = info_setting['hair_color']
    mouth = info_setting['mouth']
    eyes = info_setting['eyes']
    head = info_setting['head']

    input_info_one = [BACKGROUND_FIXED, eyes, hair_color, head, mouth]
    input_info = np.array([input_info_one for _ in range(BATCH_SIZE)])

    img = cvae_decoder([latent_vector, input_info])
    img_np = np.array(img.numpy() * 255.0, dtype=np.uint8)
    img_np_rgb = Image.fromarray(convert_RGB_to_BGR(img_np[0]))

    setting_no = info_setting['setting_no']
    img_np_rgb.save(f'models/cvae/test_result/test_output_e{"%04d" % epoch_no}_s{"%02d" % setting_no}_v{"%02d" % latent_vector_no}.png')


def test_all_cases(cvae_decoder, epoch_no):
    """
    test CVAE decoder with all the cases of 'info_settings'

    Args:
        cvae_decoder (TensorFlow Model) : CVAE decoder
        epoch_no     (int or str)       : epoch number ('finished' for already trained model)
    """

    os.makedirs('models/cvae/test_result', exist_ok=True)
    random_latent_cnt = RANDOM_LATENT_COUNT

    latent_vectors = []
    for _ in range(random_latent_cnt):
        latent_vector = np.random.normal(0.0, 1.0, size=(BATCH_SIZE, HIDDEN_DIMS))
        latent_vectors.append(latent_vector)

    for info_setting in info_settings:
        for latent_vector_no in range(random_latent_cnt):
            test_decoder(cvae_decoder, info_setting, latent_vectors[latent_vector_no], latent_vector_no, epoch_no)


def test_all_cases_with_trained_decoder():
    """
    test CVAE decoder with all the cases of 'info_settings', with trained CVAE decoder
    """

    cvae_decoder = tf.keras.models.load_model('models/cvae/cvae_decoder')
    test_all_cases(cvae_decoder, epoch_no='finished')
