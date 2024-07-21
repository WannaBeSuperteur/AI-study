# from GAI-P2

import tensorflow as tf


IMG_WIDTH = 32   # x-axis 36 ~  68 (from left) of width  104
IMG_HEIGHT = 16  # y-axis 88 ~ 104 (from top)  of height 128


class Regression_Mouth_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # 32 -> 30 -> 28 -> 14 -> 12 -> 10 (horizontal)
        # 16 -> 14 -> 12 ->  6 ->  4 ->  2 (vertical)

        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[IMG_HEIGHT, IMG_WIDTH, 3])
        self.conv_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')

        # fully connected part
        self.dense_0 = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                             kernel_regularizer=L2, name='dense_0')

        self.dense_1 = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.06),
                                             kernel_regularizer=L2, name='dense_1')

        self.final_dense = tf.keras.layers.Dense(1, activation='sigmoid',
                                                 kernel_regularizer=L2, name='dense_final')

    def call(self, inputs, training):
        inputs_img = tf.keras.layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 3))(inputs)

        # conv + pooling part
        # 32 -> 30 -> 28 -> 14 -> 12 -> 10 (horizontal)
        # 16 -> 14 -> 12 ->  6 ->  4 ->  2 (vertical)

        outputs_0 = self.conv_0(inputs_img)  # horizontal 30 / vertical 14
        outputs_1 = self.conv_1(outputs_0)   # 28 / 12
        outputs_2 = self.pooling(outputs_1)  # 14 /  6

        outputs_3 = self.conv_2(outputs_2)   # 12 /  4
        outputs_4 = self.conv_3(outputs_3)   # 10 /  2
        outputs_flatten = self.flatten(outputs_4)

        # fully connected part
        dense = self.dense_0(outputs_flatten)  # 512
        dense = self.dropout(dense)
        dense = self.dense_1(dense)  # 128
        dense = self.dropout(dense)
        final_output = self.final_dense(dense)  # 1

        return final_output