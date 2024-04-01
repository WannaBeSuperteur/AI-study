import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import cv2


class Classify_Male_Or_Female_CNN_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # 120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11
        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[120, 120, 3])
        self.conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')

        # fully connected part
        self.dense_0 = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_0')

        self.dense_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                           kernel_regularizer=L2, name='dense_1')

        self.final_dense = tf.keras.layers.Dense(2, activation='softmax',
                                                 kernel_regularizer=L2, name='dense_final')


    def call(self, inputs, training):
        inputs = tf.keras.layers.Reshape((120, 120, 3))(inputs)

        # conv + pooling part : 120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11
        outputs_0 = self.conv_0(inputs)
        outputs_1 = self.conv_1(outputs_0)
        outputs_2 = self.pooling(outputs_1)

        outputs_3 = self.conv_2(outputs_2)
        outputs_4 = self.pooling(outputs_3)

        outputs_5 = self.conv_3(outputs_4)
        outputs_6 = self.pooling(outputs_5)
        
        outputs_7 = self.conv_4(outputs_6)
        outputs_flatten = self.flatten(outputs_7)

        # fully connected part
        dense = self.dense_0(outputs_flatten)
        dense = self.dropout(dense)
        dense = self.dense_1(dense)
        dense = self.dropout(dense)
        final_output = self.final_dense(dense)

        return final_output


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = Classify_Male_Or_Female_CNN_Model()
    return model, optimizer, early_stopping, lr_reduced


# CNN 모델 학습
def train_cnn_model(train_input, train_output):
    cnn_model, optimizer, early_stopping, lr_reduced = define_model()
    cnn_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(f'train input : {np.shape(train_input)}\n{train_input}\n')
    print(f'train output : {np.shape(train_output)}\n{train_output}\n')

    cnn_model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=15,
        validation_split=0.1
    )

    cnn_model.summary()
    return cnn_model


# 학습 데이터 로딩
def load_training_data():
    male_images_name = os.listdir('resized_images/second_dataset_male')
    female_images_name = os.listdir('resized_images/second_dataset_female')
    
    train_input = []
    train_output = []
    valid_input = []
    valid_output = []

    # for male images
    for idx, name in enumerate(male_images_name):
        if idx % 750 == 0:
            print('male', idx)
            
        img = cv2.imread('resized_images/second_dataset_male/' + name, cv2.IMREAD_UNCHANGED)
        train_input.append(np.array(img) / 255.0)
        train_output.append([1, 0])

    # for female images
    for idx, name in enumerate(female_images_name):
        if idx % 750 == 0:
            print('female', idx)
            
        img = cv2.imread('resized_images/second_dataset_female/' + name, cv2.IMREAD_UNCHANGED)
        train_input.append(np.array(img) / 255.0)
        train_output.append([0, 1])

    train_input_return = np.array(train_input)
    train_output_return = np.array(train_output)

    print(f'shape of train input : {np.shape(train_input_return)}')
    print(f'shape of train output : {np.shape(train_output_return)}')
    
    return train_input_return, train_output_return


if __name__ == '__main__':

    # 학습 데이터 받아오기
    train_input, train_output = load_training_data()    

    # CNN 모델 학습
    cnn_model = train_cnn_model(train_input, train_output)

    # CNN 모델 저장
    cnn_model.save('classify_male_or_female')
