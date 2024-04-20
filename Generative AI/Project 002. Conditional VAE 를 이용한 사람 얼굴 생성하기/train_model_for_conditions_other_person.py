import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import cv2
import pandas as pd

from helper_regression_models import load_first_1k, read_output, read_gender_prob

IMG_SIZE = 120
ORIGINAL_IMG_SIZE = IMG_SIZE
IMG_WIDTH = ORIGINAL_IMG_SIZE // 6
IMG_HEIGHT = ORIGINAL_IMG_SIZE


class Regression_Other_Person_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2), name='max_pooling')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        #  20 ->  18 ->  16 ->  8 ->  6 ->  3 ->  1 (horizontal)
        # 120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 (vertical)

        # for left side of input image
        self.conv_0L = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='conv0L')
        self.conv_1L = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='conv1L')
        self.conv_2L = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2, name='conv2L')
        self.conv_3L = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=L2, name='conv3L')

        # for right side of input image
        self.conv_0R = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='conv0R')
        self.conv_1R = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2, name='conv1R')
        self.conv_2R = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2, name='conv2R')
        self.conv_3R = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=L2, name='conv3R')

        # fully connected part
        self.dense_0 = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_0')

        self.dense_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_1')

        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid',
                                                 kernel_regularizer=L2, name='dense_final')


    def call(self, inputs, training):
        inputs_img_left, inputs_img_right, inputs_gender = tf.split(
            inputs,
            [IMG_HEIGHT * IMG_WIDTH * 3, IMG_HEIGHT * IMG_WIDTH * 3, 2],
            axis=1
        )
        
        inputs_img_left = tf.keras.layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 3))(inputs_img_left)
        inputs_img_right = tf.keras.layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 3))(inputs_img_right)
        
        # conv + pooling part :  20 ->  18 ->  16 ->  8 ->  6 ->  3 ->  1 (horizontal)
        #                       120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 (vertical)

        # for left side of input image
        outputs_0L = self.conv_0L(inputs_img_left) # horizontal 18 / vertical 118
        outputs_0L = self.dropout(outputs_0L)
        outputs_1L = self.conv_1L(outputs_0L) # 16 / 116
        outputs_1L = self.dropout(outputs_1L)
        outputs_2L = self.pooling(outputs_1L) # 8 / 58

        outputs_3L = self.conv_2L(outputs_2L) # 6 / 56
        outputs_3L = self.dropout(outputs_3L)
        outputs_4L = self.pooling(outputs_3L) # 3 / 28

        outputs_5L = self.conv_3L(outputs_4L) # 1 / 26
        outputs_5L = self.dropout(outputs_5L)
        
        outputs_flattenL = self.flatten(outputs_5L)

        # for right side of input image
        outputs_0R = self.conv_0R(inputs_img_right) # horizontal 18 / vertical 118
        outputs_0R = self.dropout(outputs_0R)
        outputs_1R = self.conv_1R(outputs_0R) # 16 / 116
        outputs_1R = self.dropout(outputs_1R)
        outputs_2R = self.pooling(outputs_1R) # 8 / 58

        outputs_3R = self.conv_2R(outputs_2R) # 6 / 56
        outputs_3R = self.dropout(outputs_3R)
        outputs_4R = self.pooling(outputs_3R) # 3 / 28

        outputs_5R = self.conv_3R(outputs_4R) # 1 / 26
        outputs_5R = self.dropout(outputs_5R)
        
        outputs_flattenR = self.flatten(outputs_5R)

        # fully connected part
        outputs_flatten = tf.keras.layers.Concatenate()([outputs_flattenL, outputs_flattenR]) # 2 x ((1 x 26) x 128) = 6.7K

        dense = self.dense_0(outputs_flatten) # 256
        dense = tf.keras.layers.Concatenate()([dense, inputs_gender]) # 256 + 2 = 258
        dense = self.dropout(dense)
        
        dense = self.dense_1(dense) # 64
        dense = tf.keras.layers.Concatenate()([dense, inputs_gender]) # 64 + 2 = 66
        dense = self.dropout(dense)
        
        final_output = self.final_dense(dense) # 1

        return final_output


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.00048, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, factor=0.25)
        
    model = Regression_Other_Person_Model()
    return model, optimizer, early_stopping, lr_reduced


# save model architecture image
def plot_model_architecture(model, img_file_name):
    try:
        tf.keras.utils.plot_model(model, to_file=f'{img_file_name}.png', show_shapes=True)
    except Exception as e:
        print(f'model architecture image file generation error : {e}')


# CNN 모델 학습
def train_cnn_model(train_input_img_left, train_input_img_right, train_input_gender_prob, train_output):
    cnn_model, optimizer, early_stopping, lr_reduced = define_model()
    cnn_model.compile(loss='mse', optimizer=optimizer)

    print(f'train input (image - left side)  : {np.shape(train_input_img_left)}\n{train_input_img_left}\n')
    print(f'train input (image - right side) : {np.shape(train_input_img_right)}\n{train_input_img_right}\n')
    print(f'train input (gender prob)        : {np.shape(train_input_gender_prob)}\n{train_input_gender_prob}\n')
    print(f'train output                     : {np.shape(train_output)}\n{train_output}\n')

    train_input_img_left_ = np.reshape(train_input_img_left, (-1, IMG_HEIGHT * IMG_WIDTH * 3))
    train_input_img_right_ = np.reshape(train_input_img_right, (-1, IMG_HEIGHT * IMG_WIDTH * 3))
    train_input_gender_prob_ = np.reshape(train_input_gender_prob, (-1, 2))
    train_input = np.concatenate((train_input_img_left_, train_input_img_right_, train_input_gender_prob_), axis=1)

    cnn_model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=40,
        validation_split=0.1
    )

    cnn_model.summary()
#    plot_model_architecture(cnn_model, 'regression_other_person_model')
    
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


# left side image  : 이미지의 (x,y) 좌표 (  0,0), ( 20,120) 을 양 끝으로 하고 x, y 축과 평행한 20 x 120 직사각형 영역
# right side image : 이미지의 (x,y) 좌표 (100,0), (100,120) 을 양 끝으로 하고 x, y 축과 평행한 20 x 120 직사각형 영역
def crop_images(all_images):
    img_count = len(all_images)
    left_side_images = np.zeros((img_count, IMG_HEIGHT, IMG_WIDTH, 3))
    right_side_images = np.zeros((img_count, IMG_HEIGHT, IMG_WIDTH, 3))
    
    x_startL = 0
    y_startL = 0
    x_startR = IMG_SIZE - IMG_WIDTH
    y_startR = 0
    
    for i in range(img_count):
        left_side_images [i] = all_images[i][y_startL : y_startL + IMG_HEIGHT, x_startL : x_startL + IMG_WIDTH]
        right_side_images[i] = all_images[i][y_startR : y_startR + IMG_HEIGHT, x_startR : x_startR + IMG_WIDTH]
#        print(right_side_images[i])

    return left_side_images, right_side_images


# 학습 데이터 로딩
def load_training_data():

    # 각 성별의 처음 1,000장의 이미지에 대한 output values 와 해당 성별의 처음 1,000장의 이미지에 대해,
    # "이미지 조합 및 그 순서가 반드시 일치" 해야 함!!
    male_images, male_img_names = load_first_1k('resized_images/second_dataset_male')
    female_images, female_img_names = load_first_1k('resized_images/second_dataset_female')
    all_images = np.concatenate((male_images, female_images), axis=0)

    # crop images
    left_side_images, right_side_images = crop_images(all_images)

    other_person_male = read_output('regression_other_person_info_male.csv')
    other_person_female = read_output('regression_other_person_info_female.csv')
    other_person_all = np.concatenate((other_person_male, other_person_female), axis=0)

    train_input_img_left = np.array(left_side_images)
    train_input_img_right = np.array(right_side_images)
    train_input_gender_prob = get_gender_prob_data(male_img_names, female_img_names)
    train_output = np.array(other_person_all)

    print(f'shape of train input (images - left side)  : {np.shape(train_input_img_left)}')
    print(f'shape of train input (images - right side) : {np.shape(train_input_img_right)}')
    print(f'shape of train input (gender prob)         : {np.shape(train_input_gender_prob)}')
    print(f'shape of train output                      : {np.shape(train_output)}')
    
    return train_input_img_left, train_input_img_right, train_input_gender_prob, train_output


# model prediction 이 평균 (또는 특정) 값으로 수렴했는지 (=학습이 전혀 안 된 상태인지) 검사
# 성능 정량 평가가 아니므로, test data 가 아닌 train data 중 일부를 이용하여 체크
def is_converged_to_avg(cnn_model, N_imgs=45):
    train_input_img_left, train_input_img_right, train_input_gender_prob, _ = load_training_data() 
    
    train_input_img_left_ = np.reshape(train_input_img_left[:N_imgs], (-1, IMG_HEIGHT * IMG_WIDTH * 3))
    train_input_img_right_ = np.reshape(train_input_img_right[:N_imgs], (-1, IMG_HEIGHT * IMG_WIDTH * 3))
    train_input_gender_prob_ = np.reshape(train_input_gender_prob[:N_imgs], (-1, 2))

    train_input = np.concatenate((train_input_img_left_, train_input_img_right_, train_input_gender_prob_), axis=1)
    print(cnn_model(train_input))


if __name__ == '__main__':

    if 'regression_other_person' in os.listdir():
        cnn_model = tf.keras.models.load_model('regression_other_person')

    else:

        # 학습 데이터 받아오기
        train_input_img_left, train_input_img_right, train_input_gender_prob, train_output = load_training_data()    

        # CNN 모델 학습
        cnn_model = train_cnn_model(train_input_img_left, train_input_img_right, train_input_gender_prob, train_output)

        # CNN 모델 저장
        cnn_model.save('regression_other_person')

    # 모델 테스트 (평균값으로 수렴 여부)
    is_converged_to_avg(cnn_model)
