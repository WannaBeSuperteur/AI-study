import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import cv2
import pandas as pd

from helper_regression_models import load_first_1k, read_output, read_gender_prob

ORIGINAL_IMG_SIZE = 120
IMG_WIDTH = 60
IMG_HEIGHT = IMG_WIDTH // 2 + 2


class Regression_Eyes_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # 60 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11 -> 9 (horizontal)
        # 32 -> 30 -> 28 -> 14 -> 12 ->  6 ->  4 -> 2 (vertical)
        
        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[IMG_HEIGHT, IMG_WIDTH, 3])
        self.conv_1 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(96, (3, 3), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')

        # fully connected part
        self.dense_0 = tf.keras.layers.Dense(320, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_0')

        self.dense_1 = tf.keras.layers.Dense(120, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_1')

        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid',
                                                 kernel_regularizer=L2, name='dense_final')


    def call(self, inputs, training):
        inputs_img, inputs_gender = tf.split(inputs, [IMG_HEIGHT * IMG_WIDTH * 3, 2], axis=1)
        inputs_img = tf.keras.layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 3))(inputs_img)

        # conv + pooling part : 60 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11 -> 9 (horizontal)
        #                       32 -> 30 -> 28 -> 14 -> 12 ->  6 ->  4 -> 2 (vertical)
        
        outputs_0 = self.conv_0(inputs_img) # horizontal 58 / vertical 30
        outputs_1 = self.conv_1(outputs_0) # 56 / 28
        outputs_2 = self.pooling(outputs_1) # 28 / 14

        outputs_3 = self.conv_2(outputs_2) # 26 / 12
        outputs_4 = self.pooling(outputs_3) # 13 / 6

        outputs_5 = self.conv_3(outputs_4) # 11 / 4
        outputs_6 = self.conv_4(outputs_5) # 9 / 2
        outputs_flatten = self.flatten(outputs_6)

        # fully connected part
        dense = self.dense_0(outputs_flatten) # 256
        dense = tf.keras.layers.Concatenate()([dense, inputs_gender]) # 256 + 2 = 258
        
        dense = self.dropout(dense)
        dense = self.dense_1(dense) # 64
        dense = self.dropout(dense)
        final_output = self.final_dense(dense) # 1

        return final_output


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=7)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=3)
        
    model = Regression_Eyes_Model()
    return model, optimizer, early_stopping, lr_reduced


# CNN 모델 학습
def train_cnn_model(train_input_img, train_input_gender_prob, train_output):
    cnn_model, optimizer, early_stopping, lr_reduced = define_model()
    cnn_model.compile(loss='mse', optimizer=optimizer)

    print(f'train input (image)       : {np.shape(train_input_img)}\n{train_input_img}\n')
    print(f'train input (gender prob) : {np.shape(train_input_gender_prob)}\n{train_input_gender_prob}\n')
    print(f'train output              : {np.shape(train_output)}\n{train_output}\n')

    train_input_img_ = np.reshape(train_input_img, (-1, IMG_HEIGHT * IMG_WIDTH * 3))
    train_input_gender_prob_ = np.reshape(train_input_gender_prob, (-1, 2))
    train_input = np.concatenate((train_input_img_, train_input_gender_prob_), axis=1)

    cnn_model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=40,
        validation_split=0.1
    )

    cnn_model.summary()
    return cnn_model


# gender prob input data
def get_gender_prob_data(male_img_names, female_img_names):
    gender_prob = read_gender_prob()

    # male image 1K + female image 1K 에 대한 gender prob
    gender_prob_2k = []

    for name in male_img_names:
        gender_prob_of_img = gender_prob['resized_images/second_dataset_male/' + name]
        gender_prob_2k.append([gender_prob_of_img['prob_male'], gender_prob_of_img['prob_female']])

    for name in female_img_names:
        gender_prob_of_img = gender_prob['resized_images/second_dataset_female/' + name]
        gender_prob_2k.append([gender_prob_of_img['prob_male'], gender_prob_of_img['prob_female']])

    return np.array(gender_prob_2k)


# 이미지를 가로 4등분, 세로 4등분하여 16등분했을 때,
# 가로로 위쪽에서 2, 3번째 줄과 세로로 위쪽에서 2번째 줄 + (3번째 줄의 가장 위 2 pixels) 의 교차점
# (16등분 중 총 2개 영역 + 2 pixels 추가 영역) 에 해당하는 부분만 있는 이미지로 crop
def crop_images(all_images):
    img_count = len(all_images)
    cropped_all_images = np.zeros((img_count, IMG_HEIGHT, IMG_WIDTH, 3))
    
    x_start = (ORIGINAL_IMG_SIZE - IMG_WIDTH) // 2
    y_start = ORIGINAL_IMG_SIZE // 4
    
    for i in range(img_count):
        cropped_all_images[i] = all_images[i][y_start : y_start+IMG_HEIGHT, x_start : x_start+IMG_WIDTH]

    return cropped_all_images


# 학습 데이터 로딩
def load_training_data():

    # 각 성별의 처음 1,000장의 이미지에 대한 output values 와 해당 성별의 처음 1,000장의 이미지에 대해,
    # "이미지 조합 및 그 순서가 반드시 일치" 해야 함!!
    male_images, male_img_names = load_first_1k('resized_images/second_dataset_male')
    female_images, female_img_names = load_first_1k('resized_images/second_dataset_female')
    all_images = np.concatenate((male_images, female_images), axis=0)

    # crop images
    cropped_all_images = crop_images(all_images)

    # read output data
    eyes_male = read_output('regression_eyes_info_male.csv')
    eyes_female = read_output('regression_eyes_info_female.csv')
    all_eyes_info = np.concatenate((eyes_male, eyes_female), axis=0)

    # create dataset
    train_input_img = np.array(cropped_all_images)
    train_input_gender_prob = get_gender_prob_data(male_img_names, female_img_names)
    train_output = np.array(all_eyes_info)

    print(f'shape of train input (images)      : {np.shape(train_input_img)}')
    print(f'shape of train input (gender prob) : {np.shape(train_input_gender_prob)}')
    print(f'shape of train output              : {np.shape(train_output)}')
    
    return train_input_img, train_input_gender_prob, train_output


# model prediction 이 평균 (또는 특정) 값으로 수렴했는지 (=학습이 전혀 안 된 상태인지) 검사
# 성능 정량 평가가 아니므로, test data 가 아닌 train data 중 일부를 이용하여 체크
def is_converged_to_avg(cnn_model, N_imgs=15):
    train_input_img, train_input_gender_prob, _ = load_training_data() 
    
    train_input_img_ = np.reshape(train_input_img[:N_imgs], (-1, IMG_HEIGHT * IMG_WIDTH * 3))
    train_input_gender_prob_ = np.reshape(train_input_gender_prob[:N_imgs], (-1, 2))

    train_input = np.concatenate((train_input_img_, train_input_gender_prob_), axis=1)
    print(cnn_model(train_input))


if __name__ == '__main__':

    if 'regression_eyes' in os.listdir():
        cnn_model = tf.keras.models.load_model('regression_eyes')

    else:

        # 학습 데이터 받아오기
        train_input_img, train_input_gender_prob, train_output = load_training_data()    

        # CNN 모델 학습
        cnn_model = train_cnn_model(train_input_img, train_input_gender_prob, train_output)

        # CNN 모델 저장
        cnn_model.save('regression_eyes')

    # 모델 테스트 (평균값으로 수렴 여부)
    is_converged_to_avg(cnn_model)
