# from GAI-P2

import tensorflow as tf


IMG_WIDTH = 64   # x-axis 20 ~  84 (from left) of width  104
IMG_HEIGHT = 32  # y-axis 72 ~ 104 (from top)  of height 128


class Regression_Mouth_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # 64 -> 62 -> 60 -> 30 -> 28 -> 14 -> 12 -> 10 (horizontal)
        # 32 -> 30 -> 28 -> 14 -> 12 ->  6 ->  4 ->  2 (vertical)

        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[IMG_HEIGHT, IMG_WIDTH, 3])
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
        inputs_img = tf.keras.layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 3))(inputs)

        # conv + pooling part
        # 64 -> 62 -> 60 -> 30 -> 28 -> 14 -> 12 -> 10 (horizontal)
        # 32 -> 30 -> 28 -> 14 -> 12 ->  6 ->  4 ->  2 (vertical)

        outputs_0 = self.conv_0(inputs_img)  # horizontal 62 / vertical 30
        outputs_1 = self.conv_1(outputs_0)   # 60 / 28
        outputs_2 = self.pooling(outputs_1)  # 30 / 14

        outputs_3 = self.conv_2(outputs_2)   # 28 / 12
        outputs_4 = self.pooling(outputs_3)  # 14 /  6

        outputs_5 = self.conv_3(outputs_4)   # 12 /  4
        outputs_6 = self.conv_4(outputs_5)   # 10 /  2
        outputs_flatten = self.flatten(outputs_6)

        # fully connected part
        dense = self.dense_0(outputs_flatten)  # 256
        dense = self.dropout(dense)
        dense = self.dense_1(dense)  # 64
        dense = self.dropout(dense)
        final_output = self.final_dense(dense)  # 1

        return final_output