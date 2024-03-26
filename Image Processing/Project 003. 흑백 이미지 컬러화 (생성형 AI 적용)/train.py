import pandas as pd
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import mean_squared_error
import keras.backend as K

import os
import cv2

from test_helper import create_hsv_image, create_lv_1_2_3_images

INPUT_IMG_SIZE = 112
COLORIZE_MAP_SIZE = INPUT_IMG_SIZE // 8
TOTAL_PIXELS = INPUT_IMG_SIZE * INPUT_IMG_SIZE

HIDDEN_DIMS = 120
BATCH_SIZE = 32
MSE_LOSS_WEIGHT_CONSTANT = 100.0

NUM_CLASSES = 16



# random normal noise maker for VAE 
def noise_maker(noise_args):
    noise_mean = noise_args[0]
    noise_log_var = noise_args[1]
        
    noise = K.random_normal(shape=(BATCH_SIZE, HIDDEN_DIMS), mean=0.0, stddev=1.0)
    return K.exp(noise_log_var / 2.0) * noise + noise_mean


# ref-1: https://www.kaggle.com/code/mersico/cvae-from-scratch
# ref-2: https://github.com/ekzhang/vae-cnn-mnist/blob/master/MNIST%20Convolutional%20VAE%20with%20Label%20Input.ipynb
class Main_Model:
    def vae_entire_loss(self, x, y):
        x_reshaped = K.reshape(x, shape=(BATCH_SIZE, 2))
        y_reshaped = K.reshape(y, shape=(BATCH_SIZE, 2))
        mse_loss = mean_squared_error(x_reshaped, y_reshaped)
        
        kl_loss = -0.5 * K.sum(1 + self.latent_log_var - K.square(self.latent_mean) - K.exp(self.latent_log_var), axis=-1)
        return MSE_LOSS_WEIGHT_CONSTANT * mse_loss + kl_loss


    def __init__(self, dropout_rate=0.45):

        # 공통 레이어
        self.flatten = tf.keras.layers.Flatten()
        L2 = tf.keras.regularizers.l2(0.001)

        # encoder 용 레이어
        # level 0 : 이미지 전체
        self.encoder_lvl0_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv0_0')
        self.encoder_lvl0_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv0_1')
        self.encoder_lvl0_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv0_2')
        self.encoder_lvl0_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv0_3')
        self.encoder_lvl0_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv0_4')

        self.encoder_lvl0_5 = layers.Dense(96, activation='relu', name='en_lv0_5')
        self.encoder_lvl0_6 = layers.Dense(32, activation='relu', name='en_lv0_6')
        
        # level 1 : 주변 56 x 56 영역 (상하좌우 +24 pixels)
        self.encoder_lvl1_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv1_0')
        self.encoder_lvl1_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv1_1')
        self.encoder_lvl1_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv1_2')
        self.encoder_lvl1_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv1_3')
        self.encoder_lvl1_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv1_4')

        self.encoder_lvl1_5 = layers.Dense(96, activation='relu', name='en_lv1_5')
        self.encoder_lvl1_6 = layers.Dense(32, activation='relu', name='en_lv1_6')

        # level 2 : 주변 28 x 28 영역 (상하좌우 +10 pixels)
        self.encoder_lvl2_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv2_0')
        self.encoder_lvl2_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv2_1')
        self.encoder_lvl2_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv2_2')
        self.encoder_lvl2_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv2_3')
        self.encoder_lvl2_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv2_4')

        self.encoder_lvl2_5 = layers.Dense(96, activation='relu', name='en_lv2_5')
        self.encoder_lvl2_6 = layers.Dense(32, activation='relu', name='en_lv2_6')

        # level 3 : 주변 14 x 14 영역 (상하좌우 +3 pixels)
        self.encoder_lvl3_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv3_0')
        self.encoder_lvl3_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv3_1')
        self.encoder_lvl3_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv3_2')
        self.encoder_lvl3_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv3_3')
        self.encoder_lvl3_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='en_lv3_4')

        self.encoder_lvl3_5 = layers.Dense(96, activation='relu', name='en_lv3_5')
        self.encoder_lvl3_6 = layers.Dense(32, activation='relu', name='en_lv3_6')
        
        self.encoder_final_dense = layers.Dense(64, activation='relu', name='en_final')

        # decoder 용 레이어
        # level 0 : 이미지 전체
        self.decoder_lvl0_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv0_0')
        self.decoder_lvl0_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv0_1')
        self.decoder_lvl0_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv0_2')
        self.decoder_lvl0_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv0_3')
        self.decoder_lvl0_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv0_4')

        self.decoder_lvl0_5 = layers.Dense(96, activation='relu', name='de_lv0_5')
        self.decoder_lvl0_6 = layers.Dense(32, activation='relu', name='de_lv0_6')
        
        # level 1 : 주변 56 x 56 영역 (상하좌우 +24 pixels)
        self.decoder_lvl1_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv1_0')
        self.decoder_lvl1_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv1_1')
        self.decoder_lvl1_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv1_2')
        self.decoder_lvl1_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv1_3')
        self.decoder_lvl1_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv1_4')

        self.decoder_lvl1_5 = layers.Dense(96, activation='relu', name='de_lv1_5')
        self.decoder_lvl1_6 = layers.Dense(32, activation='relu', name='de_lv1_6')

        # level 2 : 주변 28 x 28 영역 (상하좌우 +10 pixels)
        self.decoder_lvl2_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv2_0')
        self.decoder_lvl2_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv2_1')
        self.decoder_lvl2_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv2_2')
        self.decoder_lvl2_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv2_3')
        self.decoder_lvl2_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv2_4')

        self.decoder_lvl2_5 = layers.Dense(96, activation='relu', name='de_lv2_5')
        self.decoder_lvl2_6 = layers.Dense(32, activation='relu', name='de_lv2_6')

        # level 3 : 주변 14 x 14 영역 (상하좌우 +3 pixels)
        self.decoder_lvl3_0 = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv3_0')
        self.decoder_lvl3_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv3_1')
        self.decoder_lvl3_2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv3_2')
        self.decoder_lvl3_3 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv3_3')
        self.decoder_lvl3_4 = layers.Conv2D(48, (3, 3), activation='relu', kernel_regularizer=L2, name='de_lv3_4')

        self.decoder_lvl3_5 = layers.Dense(96, activation='relu', name='de_lv3_5')
        self.decoder_lvl3_6 = layers.Dense(32, activation='relu', name='de_lv3_6')

        # latent vector (120) 과 결합하여 (120 + 32 + 32 + 32 + 32 = 248) 으로 만들기!
        self.decoder_dense = layers.Dense(96, activation='relu', name='de_dense')
        self.decoder_dense_final = layers.Dense(2, activation='tanh', name='de_final')
        
        # encoder (input and flattening)
        input_image_lv0 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='en_lv0_input')
        input_image_lv1 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='en_lv1_input')
        input_image_lv2 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='en_lv2_input')
        input_image_lv3 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='en_lv3_input')

        input_image_lv0_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='en_lv0_reshape')(input_image_lv0)
        input_image_lv1_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='en_lv1_reshape')(input_image_lv1)
        input_image_lv2_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='en_lv2_reshape')(input_image_lv2)
        input_image_lv3_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='en_lv3_reshape')(input_image_lv3)
        
        # encoder (conv + dense layers)

        # level 0
        input_image_lv0_0 = self.encoder_lvl0_0(input_image_lv0_)
        input_image_lv0_0 = layers.Dropout(rate=dropout_rate)(input_image_lv0_0)
        input_image_lv0_1 = self.encoder_lvl0_1(input_image_lv0_0)
        input_image_lv0_1 = layers.Dropout(rate=dropout_rate)(input_image_lv0_1)
        input_image_lv0_2 = self.encoder_lvl0_2(input_image_lv0_1)
        input_image_lv0_2 = layers.Dropout(rate=dropout_rate)(input_image_lv0_2)
        input_image_lv0_3 = self.encoder_lvl0_3(input_image_lv0_2)
        input_image_lv0_3 = layers.Dropout(rate=dropout_rate)(input_image_lv0_3)
        input_image_lv0_4 = self.encoder_lvl0_4(input_image_lv0_3)
        input_image_lv0_4 = layers.Dropout(rate=dropout_rate)(input_image_lv0_4)

        input_image_lv0_4 = layers.Flatten(name='en_lv0_flatten')(input_image_lv0_4)

        input_image_lv0_5 = self.encoder_lvl0_5(input_image_lv0_4)
        input_image_lv0_5 = layers.Dropout(rate=dropout_rate)(input_image_lv0_5)
        input_image_lv0_6 = self.encoder_lvl0_6(input_image_lv0_5)

        # level 1
        input_image_lv1_0 = self.encoder_lvl1_0(input_image_lv1_)
        input_image_lv1_0 = layers.Dropout(rate=dropout_rate)(input_image_lv1_0)
        input_image_lv1_1 = self.encoder_lvl1_1(input_image_lv1_0)
        input_image_lv1_1 = layers.Dropout(rate=dropout_rate)(input_image_lv1_1)
        input_image_lv1_2 = self.encoder_lvl1_2(input_image_lv1_1)
        input_image_lv1_2 = layers.Dropout(rate=dropout_rate)(input_image_lv1_2)
        input_image_lv1_3 = self.encoder_lvl1_3(input_image_lv1_2)
        input_image_lv1_3 = layers.Dropout(rate=dropout_rate)(input_image_lv1_3)
        input_image_lv1_4 = self.encoder_lvl1_4(input_image_lv1_3)
        input_image_lv1_4 = layers.Dropout(rate=dropout_rate)(input_image_lv1_4)

        input_image_lv1_4 = layers.Flatten(name='en_lv1_flatten')(input_image_lv1_4)

        input_image_lv1_5 = self.encoder_lvl1_5(input_image_lv1_4)
        input_image_lv1_5 = layers.Dropout(rate=dropout_rate)(input_image_lv1_5)
        input_image_lv1_6 = self.encoder_lvl1_6(input_image_lv1_5)

        # level 2
        input_image_lv2_0 = self.encoder_lvl2_0(input_image_lv2_)
        input_image_lv2_0 = layers.Dropout(rate=dropout_rate)(input_image_lv2_0)
        input_image_lv2_1 = self.encoder_lvl2_1(input_image_lv2_0)
        input_image_lv2_1 = layers.Dropout(rate=dropout_rate)(input_image_lv2_1)
        input_image_lv2_2 = self.encoder_lvl2_2(input_image_lv2_1)
        input_image_lv2_2 = layers.Dropout(rate=dropout_rate)(input_image_lv2_2)
        input_image_lv2_3 = self.encoder_lvl2_3(input_image_lv2_2)
        input_image_lv2_3 = layers.Dropout(rate=dropout_rate)(input_image_lv2_3)
        input_image_lv2_4 = self.encoder_lvl2_4(input_image_lv2_3)
        input_image_lv2_4 = layers.Dropout(rate=dropout_rate)(input_image_lv2_4)

        input_image_lv2_4 = layers.Flatten(name='en_lv2_flatten')(input_image_lv2_4)

        input_image_lv2_5 = self.encoder_lvl2_5(input_image_lv2_4)
        input_image_lv2_5 = layers.Dropout(rate=dropout_rate)(input_image_lv2_5)
        input_image_lv2_6 = self.encoder_lvl2_6(input_image_lv2_5)

        # level 3
        input_image_lv3_0 = self.encoder_lvl3_0(input_image_lv3_)
        input_image_lv3_0 = layers.Dropout(rate=dropout_rate)(input_image_lv3_0)
        input_image_lv3_1 = self.encoder_lvl3_1(input_image_lv3_0)
        input_image_lv3_1 = layers.Dropout(rate=dropout_rate)(input_image_lv3_1)
        input_image_lv3_2 = self.encoder_lvl3_2(input_image_lv3_1)
        input_image_lv3_2 = layers.Dropout(rate=dropout_rate)(input_image_lv3_2)
        input_image_lv3_3 = self.encoder_lvl3_3(input_image_lv3_2)
        input_image_lv3_3 = layers.Dropout(rate=dropout_rate)(input_image_lv3_3)
        input_image_lv3_4 = self.encoder_lvl3_4(input_image_lv3_3)
        input_image_lv3_4 = layers.Dropout(rate=dropout_rate)(input_image_lv3_4)

        input_image_lv3_4 = layers.Flatten(name='en_lv3_flatten')(input_image_lv3_4)

        input_image_lv3_5 = self.encoder_lvl3_5(input_image_lv3_4)
        input_image_lv3_5 = layers.Dropout(rate=dropout_rate)(input_image_lv3_5)
        input_image_lv3_6 = self.encoder_lvl3_6(input_image_lv3_5)

        # encoder (final)
        input_image_merged = layers.concatenate([input_image_lv0_6, input_image_lv1_6, input_image_lv2_6, input_image_lv3_6])
        input_image_final_dense = self.encoder_final_dense(input_image_merged)

        # latent space
        self.latent_mean = layers.Dense(HIDDEN_DIMS, name='lm')(input_image_final_dense)
        self.latent_log_var = layers.Dense(HIDDEN_DIMS, name='llv')(input_image_final_dense)
        self.latent_space = layers.Lambda(noise_maker, output_shape=(HIDDEN_DIMS,), name='ls')([self.latent_mean, self.latent_log_var])

        # decoder (input and flattening)
        de_input_image_lv0 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='de_lv0_input')
        de_input_image_lv1 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='de_lv1_input')
        de_input_image_lv2 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='de_lv2_input')
        de_input_image_lv3 = layers.Input(batch_shape=(BATCH_SIZE, COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE), name='de_lv3_input')

        de_input_image_lv0_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='de_lv0_reshape')(de_input_image_lv0)
        de_input_image_lv1_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='de_lv1_reshape')(de_input_image_lv1)
        de_input_image_lv2_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='de_lv2_reshape')(de_input_image_lv2)
        de_input_image_lv3_ = layers.Reshape((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE, 1), name='de_lv3_reshape')(de_input_image_lv3)
        
        latent_for_decoder = layers.Input(shape=(HIDDEN_DIMS,))

        # decoder (conv + dense layers)

        # level 0
        de_input_image_lv0_0 = self.decoder_lvl0_0(de_input_image_lv0_)
        de_input_image_lv0_0 = layers.Dropout(rate=dropout_rate)(de_input_image_lv0_0)
        de_input_image_lv0_1 = self.decoder_lvl0_1(de_input_image_lv0_0)
        de_input_image_lv0_1 = layers.Dropout(rate=dropout_rate)(de_input_image_lv0_1)
        de_input_image_lv0_2 = self.decoder_lvl0_2(de_input_image_lv0_1)
        de_input_image_lv0_2 = layers.Dropout(rate=dropout_rate)(de_input_image_lv0_2)
        de_input_image_lv0_3 = self.decoder_lvl0_3(de_input_image_lv0_2)
        de_input_image_lv0_3 = layers.Dropout(rate=dropout_rate)(de_input_image_lv0_3)
        de_input_image_lv0_4 = self.decoder_lvl0_4(de_input_image_lv0_3)
        de_input_image_lv0_4 = layers.Dropout(rate=dropout_rate)(de_input_image_lv0_4)

        de_input_image_lv0_4 = layers.Flatten(name='de_lv0_flatten')(de_input_image_lv0_4)

        de_input_image_lv0_5 = self.decoder_lvl0_5(de_input_image_lv0_4)
        de_input_image_lv0_5 = layers.Dropout(rate=dropout_rate)(de_input_image_lv0_5)
        de_input_image_lv0_6 = self.decoder_lvl0_6(de_input_image_lv0_5)

        # level 1
        de_input_image_lv1_0 = self.decoder_lvl1_0(de_input_image_lv1_)
        de_input_image_lv1_0 = layers.Dropout(rate=dropout_rate)(de_input_image_lv1_0)
        de_input_image_lv1_1 = self.decoder_lvl1_1(de_input_image_lv1_0)
        de_input_image_lv1_1 = layers.Dropout(rate=dropout_rate)(de_input_image_lv1_1)
        de_input_image_lv1_2 = self.decoder_lvl1_2(de_input_image_lv1_1)
        de_input_image_lv1_2 = layers.Dropout(rate=dropout_rate)(de_input_image_lv1_2)
        de_input_image_lv1_3 = self.decoder_lvl1_3(de_input_image_lv1_2)
        de_input_image_lv1_3 = layers.Dropout(rate=dropout_rate)(de_input_image_lv1_3)
        de_input_image_lv1_4 = self.decoder_lvl1_4(de_input_image_lv1_3)
        de_input_image_lv1_4 = layers.Dropout(rate=dropout_rate)(de_input_image_lv1_4)

        de_input_image_lv1_4 = layers.Flatten(name='de_lv1_flatten')(de_input_image_lv1_4)

        de_input_image_lv1_5 = self.decoder_lvl1_5(de_input_image_lv1_4)
        de_input_image_lv1_5 = layers.Dropout(rate=dropout_rate)(de_input_image_lv1_5)
        de_input_image_lv1_6 = self.decoder_lvl1_6(de_input_image_lv1_5)

        # level 2
        de_input_image_lv2_0 = self.decoder_lvl2_0(de_input_image_lv2_)
        de_input_image_lv2_0 = layers.Dropout(rate=dropout_rate)(de_input_image_lv2_0)
        de_input_image_lv2_1 = self.decoder_lvl2_1(de_input_image_lv2_0)
        de_input_image_lv2_1 = layers.Dropout(rate=dropout_rate)(de_input_image_lv2_1)
        de_input_image_lv2_2 = self.decoder_lvl2_2(de_input_image_lv2_1)
        de_input_image_lv2_2 = layers.Dropout(rate=dropout_rate)(de_input_image_lv2_2)
        de_input_image_lv2_3 = self.decoder_lvl2_3(de_input_image_lv2_2)
        de_input_image_lv2_3 = layers.Dropout(rate=dropout_rate)(de_input_image_lv2_3)
        de_input_image_lv2_4 = self.decoder_lvl2_4(de_input_image_lv2_3)
        de_input_image_lv2_4 = layers.Dropout(rate=dropout_rate)(de_input_image_lv2_4)

        de_input_image_lv2_4 = layers.Flatten(name='de_lv2_flatten')(de_input_image_lv2_4)

        de_input_image_lv2_5 = self.decoder_lvl2_5(de_input_image_lv2_4)
        de_input_image_lv2_5 = layers.Dropout(rate=dropout_rate)(de_input_image_lv2_5)
        de_input_image_lv2_6 = self.decoder_lvl2_6(de_input_image_lv2_5)

        # level 3
        de_input_image_lv3_0 = self.decoder_lvl3_0(de_input_image_lv3_)
        de_input_image_lv3_0 = layers.Dropout(rate=dropout_rate)(de_input_image_lv3_0)
        de_input_image_lv3_1 = self.decoder_lvl3_1(de_input_image_lv3_0)
        de_input_image_lv3_1 = layers.Dropout(rate=dropout_rate)(de_input_image_lv3_1)
        de_input_image_lv3_2 = self.decoder_lvl3_2(de_input_image_lv3_1)
        de_input_image_lv3_2 = layers.Dropout(rate=dropout_rate)(de_input_image_lv3_2)
        de_input_image_lv3_3 = self.decoder_lvl3_3(de_input_image_lv3_2)
        de_input_image_lv3_3 = layers.Dropout(rate=dropout_rate)(de_input_image_lv3_3)
        de_input_image_lv3_4 = self.decoder_lvl3_4(de_input_image_lv3_3)
        de_input_image_lv3_4 = layers.Dropout(rate=dropout_rate)(de_input_image_lv3_4)

        de_input_image_lv3_4 = layers.Flatten(name='de_lv3_flatten')(de_input_image_lv3_4)

        de_input_image_lv3_5 = self.decoder_lvl3_5(de_input_image_lv3_4)
        de_input_image_lv3_5 = layers.Dropout(rate=dropout_rate)(de_input_image_lv3_5)
        de_input_image_lv3_6 = self.decoder_lvl3_6(de_input_image_lv3_5)

        # decoder (final)
        de_input_image_merged = layers.concatenate([de_input_image_lv0_6, de_input_image_lv1_6, de_input_image_lv2_6, de_input_image_lv3_6, latent_for_decoder])

        de_input_image_dense = self.decoder_dense(de_input_image_merged)
        de_input_image_dense = layers.Dropout(rate=dropout_rate)(de_input_image_dense)        
        de_input_image_final_dense = self.decoder_dense_final(de_input_image_dense)

        # define encoder, decoder and cvae model
        self.encoder = tf.keras.Model([input_image_lv0, input_image_lv1, input_image_lv2, input_image_lv3], self.latent_space, name='encoder')
        self.decoder = tf.keras.Model([latent_for_decoder, de_input_image_lv0, de_input_image_lv1, de_input_image_lv2, de_input_image_lv3],
                                      de_input_image_final_dense, name='decoder')

        self.vae = tf.keras.Model(
            inputs=[
                input_image_lv0, input_image_lv1, input_image_lv2, input_image_lv3,
                de_input_image_lv0, de_input_image_lv1, de_input_image_lv2, de_input_image_lv3
            ],
            outputs=self.decoder([
                self.encoder([
                    input_image_lv0, input_image_lv1, input_image_lv2, input_image_lv3
                ]),
                de_input_image_lv0, de_input_image_lv1, de_input_image_lv2, de_input_image_lv3
            ]),
            name='final_vae'
        )


    def call(self, inputs, training):
        return self.vae(inputs)


# compute this project's own saturation using max(R, G, B) - min(R, G, B)
def compute_own_saturation(image):
    saturation = np.zeros((INPUT_IMG_SIZE, INPUT_IMG_SIZE))

    for i in range(INPUT_IMG_SIZE):
        for j in range(INPUT_IMG_SIZE):
            saturation[i][j] = max(image[i][j]) - min(image[i][j])

    return saturation


# 이미지에서 색상 및 채도 부분 분리해서 readme.md 에서 설명한, 색상과 채도를 나타내는 (x, y) 값으로 반환
# saturation is proportion to max(R, G, B) - min(R, G, B)
def get_hue_and_saturation(image):
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_image = image_HSV[:, :, 0]
    saturation_image = compute_own_saturation(image)
    
    return hue_image, saturation_image


# 입력 이미지 (greyscale) 만들기
def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# hue, saturation을 x, y로 변환하기
def convert_hue_saturation_to_coord(hue, saturation):
    hue_angle = (2.0 * math.pi) * (hue / 180.0)
    sat_float = saturation / 255.0

    x = sat_float * math.cos(hue_angle)
    y = sat_float * math.sin(hue_angle)

#    print(hue, saturation, hue_angle, math.cos(hue_angle), math.sin(hue_angle), sat_float, x, y)
    
    return x, y


# images 디렉토리에서 학습 데이터 추출
def create_train_and_valid_data(limit=None):
    images = os.listdir('images/')

    if limit is not None:
        img_count = min(limit, len(images))
    else:
        img_count = len(images)

    current_count = 0

    train_input_lv0 = [] # resize 112 x 112 entire image -> 14 x 14
    train_input_lv1 = [] # resize 56 x 56 area -> 14 x 14
    train_input_lv2 = [] # resize 28 x 28 area -> 14 x 14
    train_input_lv3 = [] # 14 x 14 area
    
    train_x_coord = []
    train_y_coord = []
    
    for image_name in images:
        if current_count < img_count:
            if current_count % 10 == 0:
                print(current_count)

            # read images
            image = cv2.imread('images/' + image_name, cv2.IMREAD_UNCHANGED)

            try:
                greyscale_image = np.array(get_greyscale(image))
            except:
                continue
            
            # compute hue and saturation
            hue_image, saturation_image = get_hue_and_saturation(image)

            if current_count == 0:
                print('\nhue image example (first image) :')
                print(hue_image)
                print('\nsaturation image example (first image) :')
                print(saturation_image)

            # compute x and y coord
            coord_x_all = np.zeros((INPUT_IMG_SIZE, INPUT_IMG_SIZE))
            coord_y_all = np.zeros((INPUT_IMG_SIZE, INPUT_IMG_SIZE))

            for i in range(INPUT_IMG_SIZE):
                for j in range(INPUT_IMG_SIZE):
                    x, y = convert_hue_saturation_to_coord(hue_image[i][j], saturation_image[i][j])
                    coord_x_all[i][j] = x
                    coord_y_all[i][j] = y

            coord_x = np.zeros((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE))
            coord_y = np.zeros((COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE))

            # 8 x 8 영역별 평균으로 coord_x, coord_y 계산 (112 x 112 이미지에 대해 각각 14 x 14)
            for i in range(COLORIZE_MAP_SIZE):
                for j in range(COLORIZE_MAP_SIZE):
                    i_start = i * 8
                    i_end = (i + 1) * 8
                    j_start = j * 8
                    j_end = (j + 1) * 8

                    coord_x[i][j] = coord_x_all[i_start:i_end, j_start:j_end].mean()
                    coord_y[i][j] = coord_y_all[i_start:i_end, j_start:j_end].mean()

            # 8 x 8 영역별로 train input lv.0, lv.1, lv.2, lv.3 계산
            greyscale_image_resized = cv2.resize(greyscale_image, (COLORIZE_MAP_SIZE, COLORIZE_MAP_SIZE))
            
            for i in range(COLORIZE_MAP_SIZE):
                for j in range(COLORIZE_MAP_SIZE):
                    i_start = i * 8
                    j_start = j * 8

                    # lv0 (entire image -> 14 x 14)
                    train_input_lv0.append(greyscale_image_resized)

                    # lv1 (56 x 56 -> 14 x 14), lv2 (28 x 28 -> 14 x 14), lv3 (14 x 14)
                    lv1_image, lv2_image, lv3_image = create_lv_1_2_3_images(
                        image_size=INPUT_IMG_SIZE,
                        color_map_size=COLORIZE_MAP_SIZE,
                        greyscale_image=greyscale_image,
                        i_start=i_start,
                        j_start=j_start
                    )
                    
                    train_input_lv1.append(lv1_image)
                    train_input_lv2.append(lv2_image)
                    train_input_lv3.append(lv3_image)

                    # x coord, y coord
                    train_x_coord.append([coord_x[i][j]])
                    train_y_coord.append([coord_y[i][j]])

                    if current_count == 0 and i < 4 and j < 4:
                        print(f'\nfirst image name: {image_name}, i: {i}, j: {j}')
                        print('\ntrain lv0 image (of first image) :')
                        print(greyscale_image_resized)
                        print('\ntrain lv1 image (of first image) :')
                        print(lv1_image)
                        print('\ntrain lv2 image (of first image) :')
                        print(lv2_image)
                        print('\ntrain lv3 image (of first image) :')
                        print(lv3_image)
                        print('\nx, y coord (of first image) :')
                        print(coord_x[i][j], coord_y[i][j])

            # test 이미지에 대한 결과 이미지 생성 테스트
            if current_count < 30:
                hsv_array = create_hsv_image(
                    image= 0.5 + greyscale_image.reshape((-1, INPUT_IMG_SIZE, INPUT_IMG_SIZE)) / (4.0 * 255.0),
                    coord_x=coord_x,
                    coord_y=coord_y,
                    img_size=INPUT_IMG_SIZE
                )
                convert_test_image = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)

                cv2.imwrite(f'input_convert_test_result/{image_name}_original.png', image)
                cv2.imwrite(f'input_convert_test_result/{image_name}.png', convert_test_image)

            current_count += 1
            
        else:
            break

    final_img_count = len(train_input_lv0) // BATCH_SIZE * BATCH_SIZE

    train_input_lv0 = np.array(train_input_lv0)[:final_img_count]
    train_input_lv1 = np.array(train_input_lv1)[:final_img_count]
    train_input_lv2 = np.array(train_input_lv2)[:final_img_count]
    train_input_lv3 = np.array(train_input_lv3)[:final_img_count]
    train_x_coord = np.array(train_x_coord)[:final_img_count]
    train_y_coord = np.array(train_y_coord)[:final_img_count]
    
    return train_input_lv0, train_input_lv1, train_input_lv2, train_input_lv3, train_x_coord, train_y_coord


# 모델 정의 및 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    model = Main_Model(dropout_rate=0.45) # 실제 모델은 model.vae
    return model, optimizer


# model architecture image
def plot_model_architecture(model, img_file_name):
    try:
        tf.keras.utils.plot_model(model, to_file=f'{img_file_name}.png', show_shapes=True)
    except Exception as e:
        print(f'model architecture image file generation error : {e}')


# 모델 구조 표시
def show_model_summary(model_class):
    print('\n === ENCODER ===')
    model_class.encoder.summary()
    plot_model_architecture(model_class.encoder, 'encoder')

    print('\n === DECODER ===')
    model_class.decoder.summary()
    plot_model_architecture(model_class.decoder, 'decoder')

    print('\n === VAE ===')
    model_class.vae.summary()
    plot_model_architecture(model_class.vae, 'vae')


# 모델 학습 실시 및 저장
# train_input_lv0, 1, 2, 3     : 입력 greyscale 이미지
# train_x_coord, train_y_coord : readme.md 에서 설명한, 색상과 채도를 나타내기 위한 (x, y) 좌표 값
def train_model(train_input_lv0, train_input_lv1, train_input_lv2, train_input_lv3, train_x_coord, train_y_coord):

    # normalize image
    train_input_lv0_for_model = train_input_lv0 / 255.0
    train_input_lv1_for_model = train_input_lv1 / 255.0
    train_input_lv2_for_model = train_input_lv2 / 255.0
    train_input_lv3_for_model = train_input_lv3 / 255.0
    
    model_class, optimizer = define_model()
    model_class.vae.compile(loss=model_class.vae_entire_loss, optimizer=optimizer)

    train_x_coord_2d = train_x_coord.reshape((-1, 1))
    train_y_coord_2d = train_y_coord.reshape((-1, 1))
    
    train_all_coords = np.concatenate([train_x_coord_2d, train_y_coord_2d], axis=1)
    train_all_coords_ = train_all_coords.reshape((-1, 2))

    print('input lv 0 shape :', np.shape(train_input_lv0_for_model))
    print('input lv 1 shape :', np.shape(train_input_lv1_for_model))
    print('input lv 2 shape :', np.shape(train_input_lv2_for_model))
    print('input lv 3 shape :', np.shape(train_input_lv3_for_model))
    
    print('x   coords shape :', np.shape(train_x_coord))
    print('y   coords shape :', np.shape(train_y_coord))
    print('all coords shape :', np.shape(train_all_coords_))

    print('all coords       :', np.array(train_all_coords_))

    # 학습 실시
    model_class.vae.fit(
        [
            train_input_lv0_for_model, train_input_lv1_for_model, train_input_lv2_for_model, train_input_lv3_for_model,
            train_input_lv0_for_model, train_input_lv1_for_model, train_input_lv2_for_model, train_input_lv3_for_model
        ],
        train_all_coords_,
        epochs=3, # 1 for functionality test, 20 for regular training
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 모델 구조 표시
    show_model_summary(model_class)

    # 모델 저장
    model_class.encoder.save('main_vae_encoder')
    model_class.decoder.save('main_vae_decoder')
    model_class.vae.save('main_vae')
    
    return model_class.encoder, model_class.decoder, model_class.vae


# input_convert_test_result 디렉토리 생성
def create_input_convert_test_result_dir():
    try:
        os.makedirs('input_convert_test_result')
    except:
        pass


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    np.set_printoptions(linewidth=200)
    create_input_convert_test_result_dir()

    # 학습 데이터 추출 (이미지의 greyscale 이미지 + 색상, 채도 부분)
    train_input_lv0, train_input_lv1, train_input_lv2, train_input_lv3, train_x_coord, train_y_coord = create_train_and_valid_data(limit=50) # 30 for functionality test
    
    print(f'\nshape of train input lv0: {np.shape(train_input_lv0)}, first image :')
    print(train_input_lv0[0])

    print(f'\nshape of train input lv1: {np.shape(train_input_lv1)}, first image :')
    print(train_input_lv1[0])

    print(f'\nshape of train input lv2: {np.shape(train_input_lv2)}, first image :')
    print(train_input_lv2[0])

    print(f'\nshape of train input lv3: {np.shape(train_input_lv3)}, first image :')
    print(train_input_lv3[0])

    # 학습 실시 및 모델 저장
    vae_encoder, vae_decoder, vae_model = train_model(
        train_input_lv0, train_input_lv1, train_input_lv2, train_input_lv3,
        train_x_coord, train_y_coord
    )
