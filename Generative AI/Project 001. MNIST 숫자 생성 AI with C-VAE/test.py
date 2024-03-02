import tensorflow as tf
import keras.backend as K
from PIL import Image

import numpy as np
from train import noise_maker
import os

HIDDEN_DIMS = 40
NUM_CLASSES = 10
BATCH_SIZE = 32

encoder = tf.keras.models.load_model('cvae_encoder_model', custom_objects={'noise_maker': noise_maker})
decoder = tf.keras.models.load_model('cvae_decoder_model')


def create_img_dir():
    if 'test_imgs' not in os.listdir():
        os.makedirs('test_imgs')


# 이미지 생성 및 저장
def generate_image_with_num(num, latent_space=None, prefix='test_number'):
    if latent_space is None:
        latent_space = np.random.normal(0.0, 1.0, size=(1, HIDDEN_DIMS))

    input_class = np.zeros((1, NUM_CLASSES))
    input_class[0][num] = 1.0

    img = decoder([latent_space, input_class])
    img_np = img.numpy() * 255.0
    img_np_rgb = Image.fromarray(img_np[0]).convert('RGB')
    img_np_rgb.save(f'test_imgs/{prefix}_{num}.png')


if __name__ == '__main__':
    create_img_dir()

    # create number image WITHOUT latent space
    for i in range(NUM_CLASSES):
        generate_image_with_num(i)

    # create number image WITH latent space
    for i in range(NUM_CLASSES):
        for j in range(5):
            latent_space = np.random.normal(0.0, 1.0, size=(1, HIDDEN_DIMS))
            generate_image_with_num(i, latent_space=latent_space, prefix=f'test_num_with_ls_{j}')
