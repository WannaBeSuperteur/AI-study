# from GAI-P2

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from keras.losses import mean_squared_error
import keras.backend as K
import matplotlib.pyplot as plt

import cv2
from cvae_model_utils import BATCH_SIZE, TOTAL_INPUT_IMG_VALUES, MSE_LOSS_WEIGHT, TRAIN_EPOCHS
from cvae_model_utils import scheduler, save_cvae_loss_log, show_model_summary


# TODO complete model architecture
class CVAE_Model:

    def get_mse_and_kl_loss(self, x, y):
        x_reshaped = K.reshape(x, shape=(BATCH_SIZE, TOTAL_INPUT_IMG_VALUES))
        y_reshaped = K.reshape(y, shape=(BATCH_SIZE, TOTAL_INPUT_IMG_VALUES))

        mse_loss = MSE_LOSS_WEIGHT * mean_squared_error(x_reshaped, y_reshaped)
        kl_loss = -0.5 * K.sum(1 + self.latent_log_var - K.square(self.latent_mean) - K.exp(self.latent_log_var), axis=-1)

        return mse_loss, kl_loss, y_reshaped


    # VAE 의 loss function
    def vae_loss(self, x, y):
        mse_loss, kl_loss, _ = self.get_mse_and_kl_loss(x, y)
        return mse_loss + kl_loss

    def __init__(self, dropout_rate=0.25):
        pass

        # TODO complete model architecture

    def call(self, inputs, training):
        return self.cvae(inputs)


# C-VAE 모델 정의 및 반환
def define_cvae_model():
    optimizer = optimizers.Adam(0.0006, decay=1e-6)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cvae_module = CVAE_Model(dropout_rate=0.25) # 실제 모델은 model.cvae

    return cvae_module, optimizer, scheduler_callback


# C-VAE 모델 학습 실시 및 모델 저장
# train_info = train_condition (N, 5)
def train_cvae_model(train_input, train_info):

    # to solve "You must feed a value for placeholder tensor {tensor_name} with dtype float and shape {shape}."
    tf.keras.backend.set_learning_phase(False)
    cvae_module, optimizer, scheduler_callback = define_cvae_model()
    cvae_module.cvae.compile(loss=cvae_module.vae_loss, optimizer=optimizer)

    # 학습 실시
    train_history = cvae_module.cvae.fit(
        [train_input, train_info, train_info], train_input,
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[scheduler_callback],
        shuffle=True
    )
    save_cvae_loss_log(train_history)

    # 모델 구조 표시
    show_model_summary(cvae_module)

    cvae_module.encoder.save('models/cvae/cvae_encoder')
    cvae_module.decoder.save('models/cvae/cvae_decoder')
    cvae_module.cvae.save('models/cvae/cvae_entire')

    return cvae_module.encoder, cvae_module.decoder, cvae_module.cvae