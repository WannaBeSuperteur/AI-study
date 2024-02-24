import pandas as pd
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class CNN_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv part
        self.conv_00 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=[128, 128, 3], name='conv_00')
        self.conv_01 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_01')

        self.conv_10 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same', name='conv_10')
        self.conv_11 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same', name='conv_11')

        self.conv_20 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_20')
        self.conv_21 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_21')

        self.conv_30 = tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same', name='conv_30')
        self.conv_31 = tf.keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same', name='conv_31')

        self.conv_40 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_40')
        self.conv_41 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_41')
        
        # fully connected part
        self.dense_0 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                             kernel_regularizer=L2, name='dense_0')

        self.dense_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                             kernel_regularizer=L2, name='dense_1')

        self.dense_final = tf.keras.layers.Dense(2, activation='softmax',
                                             kernel_regularizer=L2, name='dense_final')


    def call(self, inputs, training):

        # conv part 0 (128 -> 64)
        outputs_00 = self.conv_00(inputs)
        outputs_00 = self.dropout(outputs_00)
        outputs_01 = self.conv_01(outputs_00)

        # Concatenate (?, A, B, 32) with (?, A, B, 3)
        outputs_0 = tf.keras.layers.Concatenate()([inputs, outputs_01])
        outputs_0 = self.pooling(outputs_0)

        # conv part 1 (64 -> 32)
        outputs_10 = self.conv_10(outputs_0)
        outputs_10 = self.dropout(outputs_10)
        outputs_11 = self.conv_11(outputs_10)
        
        outputs_1 = tf.keras.layers.Concatenate()([outputs_0, outputs_11])
        outputs_1 = self.pooling(outputs_1)

        # conv part 2 (32 -> 16)
        outputs_20 = self.conv_20(outputs_1)
        outputs_20 = self.dropout(outputs_20)
        outputs_21 = self.conv_21(outputs_20)
        
        outputs_2 = tf.keras.layers.Concatenate()([outputs_1, outputs_21])
        outputs_2 = self.pooling(outputs_2)

        # conv part 3 (16 -> 8)
        outputs_30 = self.conv_30(outputs_2)
        outputs_30 = self.dropout(outputs_30)
        outputs_31 = self.conv_31(outputs_30)
        
        outputs_3 = tf.keras.layers.Concatenate()([outputs_2, outputs_31])
        outputs_3 = self.pooling(outputs_3)

        # conv part 4 (8 -> 8)
        outputs_40 = self.conv_40(outputs_3)
        outputs_40 = self.dropout(outputs_40)
        outputs_41 = self.conv_41(outputs_40)

        # flatten
        outputs_4 = self.flatten(outputs_41)

        # dense
        outputs_dense_0 = self.dense_0(outputs_4)
        outputs_dense_0 = self.dropout(outputs_dense_0)
        outputs_dense_1 = self.dense_1(outputs_dense_0)
        outputs_dense_1 = self.dropout(outputs_dense_1)
        outputs_dense_final = self.dense_final(outputs_dense_1)

        return outputs_dense_final


# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = CNN_Model(dropout_rate=0.25)
    return model, optimizer, early_stopping, lr_reduced


# 모델 학습
def train_model(train_input, valid_input, train_output, valid_output):
    model, optimizer, early_stopping, lr_reduced = define_model()
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=5, # each epoch takes around 7 minutes
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('project_001_cnn_model')
    return model


# 모델 테스트
def test_model(test_input, model):
    return model(test_input)
