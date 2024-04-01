import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import cv2
import pandas as pd

from helper_regression_models import load_first_1k, read_output, read_gender_prob

IMG_SIZE = 120


class Regression_Hair_Color_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # 120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11
        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[IMG_SIZE, IMG_SIZE, 3])
        self.conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')

        # fully connected part
        self.dense_0 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_0')

        self.dense_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_1')

        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid',
                                                 kernel_regularizer=L2, name='dense_final')


    def call(self, inputs, training):
        inputs_img, inputs_gender = tf.split(inputs, [IMG_SIZE * IMG_SIZE * 3, 2], axis=1)
        inputs_img = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 3))(inputs_img)

        # conv + pooling part : 120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11
        outputs_0 = self.conv_0(inputs_img)
        outputs_1 = self.conv_1(outputs_0)
        outputs_2 = self.pooling(outputs_1)

        outputs_3 = self.conv_2(outputs_2)
        outputs_4 = self.pooling(outputs_3)

        outputs_5 = self.conv_3(outputs_4)
        outputs_6 = self.pooling(outputs_5)
        
        outputs_7 = self.conv_4(outputs_6)
        outputs_flatten = self.flatten(outputs_7)

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
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = Regression_Hair_Color_Model()
    return model, optimizer, early_stopping, lr_reduced


# CNN 모델 학습
def train_cnn_model(train_input_img, train_input_gender_prob, train_output):
    cnn_model, optimizer, early_stopping, lr_reduced = define_model()
    cnn_model.compile(loss='mse', optimizer=optimizer)

    print(f'train input (image)       : {np.shape(train_input_img)}\n{train_input_img}\n')
    print(f'train input (gender prob) : {np.shape(train_input_gender_prob)}\n{train_input_gender_prob}\n')
    print(f'train output              : {np.shape(train_output)}\n{train_output}\n')

    train_input_img_ = np.reshape(train_input_img, (-1, IMG_SIZE * IMG_SIZE * 3))
    train_input_gender_prob_ = np.reshape(train_input_gender_prob, (-1, 2))
    train_input = np.concatenate((train_input_img_, train_input_gender_prob_), axis=1)

    cnn_model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=2,
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


# 학습 데이터 로딩
def load_training_data():

    # 각 성별의 처음 1,000장의 이미지에 대한 output values 와 해당 성별의 처음 1,000장의 이미지에 대해,
    # "이미지 조합 및 그 순서가 반드시 일치" 해야 함!!
    male_images, male_img_names = load_first_1k('resized_images/second_dataset_male')
    female_images, female_img_names = load_first_1k('resized_images/second_dataset_female')
    all_images = np.concatenate((male_images, female_images), axis=0)

    hair_color_male = read_output('regression_hair_color_info_male.csv')
    hair_color_female = read_output('regression_hair_color_info_female.csv')
    all_hair_colors = np.concatenate((hair_color_male, hair_color_female), axis=0)

    train_input_img = np.array(all_images)
    train_input_gender_prob = get_gender_prob_data(male_img_names, female_img_names)
    train_output = np.array(all_hair_colors)

    print(f'shape of train input (images)      : {np.shape(train_input_img)}')
    print(f'shape of train input (gender prob) : {np.shape(train_input_gender_prob)}')
    print(f'shape of train output              : {np.shape(train_output)}')
    
    return train_input_img, train_input_gender_prob, train_output


# model prediction 이 평균 (또는 특정) 값으로 수렴했는지 (=학습이 전혀 안 된 상태인지) 검사
# 성능 정량 평가가 아니므로, test data 가 아닌 train data 중 일부를 이용하여 체크
def is_converged_to_avg(cnn_model, N_imgs=15):
    train_input_img, train_input_gender_prob, _ = load_training_data() 
    
    train_input_img_ = np.reshape(train_input_img[:N_imgs], (-1, IMG_SIZE * IMG_SIZE * 3))
    train_input_gender_prob_ = np.reshape(train_input_gender_prob[:N_imgs], (-1, 2))

    train_input = np.concatenate((train_input_img_, train_input_gender_prob_), axis=1)
    print(cnn_model(train_input))


if __name__ == '__main__':

    if 'regression_hair_color' in os.listdir():
        cnn_model = tf.keras.models.load_model('regression_hair_color')

    else:

        # 학습 데이터 받아오기
        train_input_img, train_input_gender_prob, train_output = load_training_data()    

        # CNN 모델 학습
        cnn_model = train_cnn_model(train_input_img, train_input_gender_prob, train_output)

        # CNN 모델 저장
        cnn_model.save('regression_hair_color')

    # 모델 테스트 (평균값으로 수렴 여부)
    is_converged_to_avg(cnn_model)
