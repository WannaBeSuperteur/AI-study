import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class Classify_Male_Or_Female_CNN_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # 120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11
        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[120, 120, 1])
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
        inputs = tf.keras.layers.Reshape((120, 120, 1))(inputs)

        # conv + pooling part : 120 -> 118 -> 116 -> 58 -> 56 -> 28 -> 26 -> 13 -> 11
        outputs_0 = self.conv_0(inputs)
        outputs_1 = self.conv_1(outputs_0)
        outputs_2 = self.conv_2(outputs_2)
        outputs_3 = self.pooling(outputs_2)

        outputs_4 = self.conv_3(outputs_3)
        outputs_5 = self.pooling(outputs_4)

        outputs_6 = self.conv_4(outputs_5)
        outputs_7 = self.pooling(outputs_6)
        
        outputs_8 = self.conv_5(outputs_7)
        outputs_flatten = self.flatten(outputs_8)

        # fully connected part
        dense = self.dense_0(outputs_flatten)
        dense = self.dropout(dense)
        dense = self.dense_1(dense)
        dense = self.dropout(dense)
        final_output = self.final_dense(dense)

        return final_output
