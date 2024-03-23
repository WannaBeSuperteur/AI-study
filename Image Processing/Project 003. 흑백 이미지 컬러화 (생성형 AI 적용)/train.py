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


INPUT_IMG_SIZE = 112
HIDDEN_DIMS = 60
TOTAL_PIXELS = INPUT_IMG_SIZE * INPUT_IMG_SIZE
BATCH_SIZE = 32
MSE_LOSS_WEIGHT_CONSTANT = 100.0


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
        x_reshaped = K.reshape(x, shape=(BATCH_SIZE, 2 * TOTAL_PIXELS))
        y_reshaped = K.reshape(y, shape=(BATCH_SIZE, 2 * TOTAL_PIXELS))
        mse_loss = mean_squared_error(x_reshaped, y_reshaped)
        
        kl_loss = -0.5 * K.sum(1 + self.latent_log_var - K.square(self.latent_mean) - K.exp(self.latent_log_var), axis=-1)
        return MSE_LOSS_WEIGHT_CONSTANT * mse_loss + kl_loss


    def __init__(self, dropout_rate=0.45):

        # 공통 레이어
        self.flatten = tf.keras.layers.Flatten()
        L2 = tf.keras.regularizers.l2(0.001)

        # encoder 용 레이어
        self.encoder_cnn0 = layers.Conv2D(16, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec0')
        self.encoder_cnn1 = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec1')
        self.encoder_cnn2 = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec2')
        self.encoder_cnn3 = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='ec3')
        
        self.encoder_dense0 = layers.Dense(256, activation='relu', name='ed0')
        self.encoder_ad0 = layers.Dense(64, activation='relu', name='ead0') # input 과 직접 연결

        # decoder 용 레이어
        # decoder mapping architecture (like U-net) 용 레이어
        self.decoder_map_e0 = layers.Conv2D(16, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_e0')
        self.decoder_map_e1 = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_e1')
        self.decoder_map_e2 = layers.Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_e2')
        self.decoder_map_e3 = layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_e3')

        # latent vector (60) 과 결합하여 (60 + 16 = 76) 으로 만든 이후 decoder_map_d0 레이어로 이동
        self.decoder_map_bottleneck = layers.Dense(16, activation='relu', name='dmap_bottleneck')
        self.decoder_map_bottleneck_restore = layers.Dense(128 * (INPUT_IMG_SIZE // 16) * (INPUT_IMG_SIZE // 16), activation='relu', name='dmap_bottleneck_restore')

        # for x coordinate
        self.decoder_map_d0_x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d0_x')
        self.decoder_map_d1_x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d1_x')
        self.decoder_map_d2_x = layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d2_x')
        self.decoder_map_d3_x = layers.Conv2DTranspose(1, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d3_x')

        # for y coordinate
        self.decoder_map_d0_y = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d0_y')
        self.decoder_map_d1_y = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d1_y')
        self.decoder_map_d2_y = layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d2_y')
        self.decoder_map_d3_y = layers.Conv2DTranspose(1, (3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=L2, name='dmap_d3_y')
        
        # encoder (main stream)
        input_image = layers.Input(batch_shape=(BATCH_SIZE, INPUT_IMG_SIZE, INPUT_IMG_SIZE))
        input_image_reshaped = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE, 1))(input_image)
        
        enc_c0 = self.encoder_cnn0(input_image_reshaped)
        enc_c0 = layers.Dropout(rate=dropout_rate)(enc_c0)
        enc_c1 = self.encoder_cnn1(enc_c0)
        enc_c1 = layers.Dropout(rate=dropout_rate)(enc_c1)
        enc_c2 = self.encoder_cnn2(enc_c1)
        enc_c2 = layers.Dropout(rate=dropout_rate)(enc_c2)
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
        input_for_decoder_ = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE, 1))(input_for_decoder)

        # decoder mapping architecture (like U-net)
        dec_e0 = self.decoder_map_e0(input_for_decoder_)
        dec_e0 = layers.Dropout(rate=dropout_rate)(dec_e0)
        dec_e1 = self.decoder_map_e1(dec_e0)
        dec_e1 = layers.Dropout(rate=dropout_rate)(dec_e1)
        dec_e2 = self.decoder_map_e2(dec_e1)
        dec_e2 = layers.Dropout(rate=dropout_rate)(dec_e2)
        dec_e3 = self.decoder_map_e3(dec_e2)

        dec_flatten = self.flatten(dec_e3)
        dec_bn = self.decoder_map_bottleneck(dec_flatten)
        dec_bn_ = layers.concatenate([dec_bn, latent_for_decoder])
        
        dec_bn_restored = self.decoder_map_bottleneck_restore(dec_bn_)
        dec_bn_restored = layers.Reshape((INPUT_IMG_SIZE // 16, INPUT_IMG_SIZE // 16, 128))(dec_bn_restored)

        # for x coord
        dec_d0_x = self.decoder_map_d0_x(dec_bn_restored)
        dec_d0_x = layers.Add()([dec_d0_x, dec_e2])
        dec_d0_x = layers.Dropout(rate=dropout_rate)(dec_d0_x)
        
        dec_d1_x = self.decoder_map_d1_x(dec_d0_x)
        dec_d1_x = layers.Add()([dec_d1_x, dec_e1])
        dec_d1_x = layers.Dropout(rate=dropout_rate)(dec_d1_x)
        
        dec_d2_x = self.decoder_map_d2_x(dec_d1_x)
        dec_d2_x = layers.Add()([dec_d2_x, dec_e0])
        dec_d2_x = layers.Dropout(rate=dropout_rate)(dec_d2_x)
        
        dec_d3_x = self.decoder_map_d3_x(dec_d2_x)

        # for y coord
        dec_d0_y = self.decoder_map_d0_y(dec_bn_restored)
        dec_d0_y = layers.Add()([dec_d0_y, dec_e2])
        dec_d0_y = layers.Dropout(rate=dropout_rate)(dec_d0_y)
        
        dec_d1_y = self.decoder_map_d1_y(dec_d0_y)
        dec_d1_y = layers.Add()([dec_d1_y, dec_e1])
        dec_d1_y = layers.Dropout(rate=dropout_rate)(dec_d1_y)
        
        dec_d2_y = self.decoder_map_d2_y(dec_d1_y)
        dec_d2_y = layers.Add()([dec_d2_y, dec_e0])
        dec_d2_y = layers.Dropout(rate=dropout_rate)(dec_d2_y)
        
        dec_d3_y = self.decoder_map_d3_y(dec_d2_y)

        # decoder merge
        dec_d3_merged = layers.concatenate([dec_d3_x, dec_d3_y], axis=3)
        dec_final_coord_x_and_y = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE, 2))(dec_d3_merged)

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
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_image = image_HSV[:, :, 0]
    saturation_image = image_HSV[:, :, 1]
    
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
        img_count = min(limit, len(images)) // BATCH_SIZE * BATCH_SIZE
    else:
        img_count = len(images) // BATCH_SIZE * BATCH_SIZE

    current_count = 0

    train_input = []
    train_x_coord = []
    train_y_coord = []
    
    for image_name in images:
        if current_count < img_count:
            if (current_count % 10 == 0 and current_count < 100) or current_count % 100 == 0:
                print(current_count)

            # read images
            image = cv2.imread('images/' + image_name, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            greyscale_image = np.array(get_greyscale(image))

            # compute hue and saturation
            hue_image, saturation_image = get_hue_and_saturation(image)

            if current_count == 0:
                print('\nhue image example (first image) :')
                print(hue_image)
                print('\nsaturation image example (first image) :')
                print(saturation_image)

            # compute x and y coord
            coord_x = np.zeros((INPUT_IMG_SIZE, INPUT_IMG_SIZE))
            coord_y = np.zeros((INPUT_IMG_SIZE, INPUT_IMG_SIZE))

            for i in range(INPUT_IMG_SIZE):
                for j in range(INPUT_IMG_SIZE):
                    x, y = convert_hue_saturation_to_coord(hue_image[i][j], saturation_image[i][j])
                    coord_x[i][j] = x
                    coord_y[i][j] = y

            # append to train data
            train_input.append(greyscale_image)
            train_x_coord.append(coord_x)
            train_y_coord.append(coord_y)

            current_count += 1
            
        else:
            break

    train_input = np.array(train_input)
    train_x_coord = np.array(train_x_coord)
    train_y_coord = np.array(train_y_coord)

    return train_input, train_x_coord, train_y_coord


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
# train_input                  : 입력 greyscale 이미지
# train_x_coord, train_y_coord : readme.md 에서 설명한, 색상과 채도를 나타내기 위한 (x, y) 좌표 값
def train_model(train_input, train_x_coord, train_y_coord):

    # normalize image
    train_input_for_model = train_input / 255.0
    
    model_class, optimizer = define_model()
    model_class.vae.compile(loss=model_class.vae_entire_loss, optimizer=optimizer)

    train_x_coord_4d = train_x_coord.reshape((-1, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 1))
    train_y_coord_4d = train_y_coord.reshape((-1, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 1))
    
    train_all_coords = np.concatenate([train_x_coord_4d, train_y_coord_4d], axis=3)

    print('input      shape :', np.shape(train_input_for_model))
    print('x   coords shape :', np.shape(train_x_coord))
    print('y   coords shape :', np.shape(train_y_coord))
    print('all coords shape :', np.shape(train_all_coords))

    # 학습 실시
    model_class.vae.fit(
        [train_input_for_model, train_input_for_model], train_all_coords,
        epochs=1, # 1 for functionality test, 20 for regular training
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


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    np.set_printoptions(linewidth=160)

    # 학습 데이터 추출 (이미지의 greyscale 이미지 + 색상, 채도 부분)
    train_input, train_x_coord, train_y_coord = create_train_and_valid_data(limit=320) # 320 for functionality test
    
    print(f'\nshape of train input: {np.shape(train_input)}, first image :')
    print(train_input[0])
    
    print(f'\nshape of train x coord: {np.shape(train_x_coord)}, first image :')
    print(train_x_coord[0])
    
    print(f'\nshape of train y coord: {np.shape(train_y_coord)}, first image :')
    print(train_y_coord[0])

    # 학습 실시 및 모델 저장
    vae_encoder, vae_decoder, vae_model = train_model(train_input, train_x_coord, train_y_coord)
