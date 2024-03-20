import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import mean_squared_error
import keras.backend as K

import os


INPUT_IMG_SIZE = 120
HIDDEN_DIMS = 60
TOTAL_PIXELS = INPUT_IMG_SIZE * INPUT_IMG_SIZE
BATCH_SIZE = 32


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
        x_reshaped = K.reshape(x, shape=(BATCH_SIZE, TOTAL_PIXELS))
        y_reshaped = K.reshape(y, shape=(BATCH_SIZE, TOTAL_PIXELS))
        mse_loss = TOTAL_CELLS * mean_squared_error(x_reshaped, y_reshaped)
        
        kl_loss = -0.5 * K.sum(1 + self.latent_log_var - K.square(self.latent_mean) - K.exp(self.latent_log_var), axis=-1)
        return mse_loss, kl_loss


    def __init__(self, dropout_rate=0.45):

        # 공통 레이어
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')

        L2 = tf.keras.regularizers.l2(0.001)

        # encoder 용 레이어
        self.encoder_cnn0 = layers.Conv2D(16, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec0')
        self.encoder_cnn1 = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec1')
        self.encoder_cnn2 = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec2')
        self.encoder_cnn3 = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec2')
        
        self.encoder_dense0 = layers.Dense(256, activation='relu', name='ed0')
        self.encoder_ad0 = layers.Dense(64, activation='relu', name='ead0') # input 과 직접 연결

        # decoder 용 레이어
        self.decoder_dense0 = layers.Dense(320, activation='relu', name='dd0')
        self.decoder_dense1 = layers.Dense(80 * TOTAL_PIXELS // (16 * 16), activation='relu', name='dd1')

        self.decoder_cnn0 = layers.Conv2DTranspose(40, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dc0')
        self.decoder_cnn1 = layers.Conv2DTranspose(40, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dc1')
        self.decoder_cnn2 = layers.Conv2DTranspose(20, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dc2')
        self.decoder_cnn3 = layers.Conv2DTranspose(2, (3, 3), strides=1, activation='relu', padding='same', kernel_regularizer=L2, name='dc2')

        # encoder (main stream)
        input_image = layers.Input(batch_shape=(BATCH_SIZE, INPUT_IMG_SIZE, INPUT_IMG_SIZE))
        input_image_reshaped = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE, 1))(input_image)
        
        enc_c0 = self.encoder_cnn0(input_image_reshaped)
        enc_c0 = self.dropout(enc_c0)
        enc_c1 = self.encoder_cnn1(enc_c0)
        enc_c1 = self.dropout(enc_c1)
        enc_c2 = self.encoder_cnn2(enc_c1)
        enc_c2 = self.dropout(enc_c2)
        enc_c3 = self.encoder_cnn3(enc_c2)
        
        enc_flatten = self.flatten(enc_c3)
        enc_d0 = self.encoder_dense0(enc_flatten)

        # encoder (additional stream)
        enc_flatten_for_ad = self.flatten(input_image)
        enc_ad0 = self.encoder_ad0(enc_flatten_for_ad)

        # encoder (concatenated)
        end_d0_ad0 = layers.concatenate([enc_d0, enc_ad0])

        # latent space
        self.latent_mean = layers.Dense(HIDDEN_DIMS, name='lm')(end_d0_ad0)
        self.latent_log_var = layers.Dense(HIDDEN_DIMS, name='llv')(end_d0_ad0)
        self.latent_space = layers.Lambda(noise_maker, output_shape=(HIDDEN_DIMS,), name='ls')([self.latent_mean, self.latent_log_var])

        # decoder
        latent_for_decoder = layers.Input(shape=(HIDDEN_DIMS,))
        input_for_decoder = layers.Input(shape=(INPUT_IMG_SIZE, INPUT_IMG_SIZE))
        input_for_decoder_flatten = input_for_decoder.flatten()

        dec_merged = layers.concatenate([latent_for_decoder, input_for_decoder_flatten])
        dec_d0 = self.decoder_dense0(dec_merged)
        dec_d1 = self.decoder_dense1(dec_d0)
        dec_reshaped = layers.Reshape((INPUT_IMG_SIZE // 16, INPUT_IMG_SIZE // 16, 160))(dec_d1)

        dec_c0 = self.decoder_cnn0(dec_reshaped)
        dec_c0 = self.dropout(dec_c0)
        dec_c1 = self.decoder_cnn1(dec_c0)
        dec_c1 = self.dropout(dec_c1)
        dec_c2 = self.decoder_cnn2(dec_c1)
        dec_c2 = self.dropout(dec_c2)
        dec_c3 = self.decoder_cnn3(dec_c2)
        
        dec_final_coord_x_and_y = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE))(dec_c3)

        # define encoder, decoder and cvae model
        self.encoder = tf.keras.Model([input_image], self.latent_space, name='encoder')
        self.decoder = tf.keras.Model([latent_for_decoder, input_for_decoder], dec_final_coord_x_and_y, name='decoder')

        self.vae = tf.keras.Model(
            inputs=[input_image, input_for_decoder],
            outputs=self.decoder([self.encoder([input_image]), input_for_decoder]),
            name='final_vae'
        )


    def call(self, inputs, training):
        return self.vae(inputs)


# 이미지에서 색상 및 채도 부분 분리해서 readme.md 에서 설명한, 색상과 채도를 나타내는 (x, y) 값으로 반환
def get_hue_and_saturation(image):
    pass


# 입력 이미지 (greyscale) 만들기
def get_greyscale(image):
    pass


# images 디렉토리에서 학습 데이터 추출
def create_train_and_valid_data(limit=None):
    images = os.listdir('images/')
    img_count = len(images) // BATCH_SIZE * BATCH_SIZE
    current_count = 0

    train_input = []
    train_x_coord = []
    train_y_coord = []
    
    for image_name in images:

        # read images
        if current_count < img_count:
            image = cv2.imread('images/' + image_name, cv2.IMREAD_UNCHANGED)
        else:
            break

    return train_input, train_x_coord, train_y_coord


# 모델 정의 및 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    model = Main_Model(dropout_rate=0.45) # 실제 모델은 model.vae
    return model, optimizer


# 모델 학습 실시 및 저장
# train_input                  : 입력 greyscale 이미지
# train_x_coord, train_y_coord : readme.md 에서 설명한, 색상과 채도를 나타내기 위한 (x, y) 좌표 값
def train_model(train_input, train_x_coord, train_y_coord, train_color_info):
    model_class, optimizer = define_cvae_model()
    model_class.vae.compile(loss=model_class.vae_entire_loss, optimizer=optimizer)

    # 학습 실시
    model_class.cvae.fit(
        [train_input, train_x_coord, train_y_coord], train_input,
        epochs=40,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print('\n === ENCODER ===')
    model_class.encoder.summary()

    print('\n === DECODER ===')
    model_class.decoder.summary()

    print('\n === C-VAE ===')
    model_class.vae.summary()

    # 모델 저장
    model_class.encoder.save('main_vae_encoder')
    model_class.decoder.save('main_vae_decoder')
    model_class.cvae.save('main_vae')
    
    return model_class.encoder, model_class.decoder, model_class.cvae


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    # 학습 데이터 추출 (이미지의 greyscale 이미지 + 색상, 채도 부분)
    train_input, train_x_coord, train_y_coord = create_train_and_valid_data()
    
    print(f'\nshape of train input: {np.shape(train_input)}')
    print(train_input)
    
    print(f'\nshape of train x coord: {np.shape(train_x_coord)}')
    print(train_x_coord)
    
    print(f'\nshape of train y coord: {np.shape(train_y_coord)}')
    print(train_y_coord)

    # 학습 실시 및 모델 저장
    cvae_encoder, cvae_decoder, cvae_model = train_cvae_model(train_input, train_x_coord, train_y_coord)
