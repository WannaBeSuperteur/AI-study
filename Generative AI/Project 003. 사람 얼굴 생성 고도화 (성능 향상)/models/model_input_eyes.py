# from GAI-P2

import tensorflow as tf

IMG_WIDTH = 56   # x-axis 24 ~ 80 (from left) of width  104
IMG_HEIGHT = 24  # y-axis 48 ~ 72 (from top)  of height 128


class Regression_Eyes_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # 56 -> 54 -> 52 -> 26 -> 24 -> 12 -> 10 (horizontal)
        # 24 -> 22 -> 20 -> 10 ->  8 ->  4 ->  2 (vertical)

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

        # conv + pooling part : 56 -> 54 -> 52 -> 26 -> 24 -> 12 -> 10 (horizontal)
        #                       24 -> 22 -> 20 -> 10 ->  8 ->  4 ->  2 (vertical)

        outputs_0 = self.conv_0(inputs_img)  # horizontal 54 / vertical 22
        outputs_1 = self.conv_1(outputs_0)   # 52 / 20
        outputs_2 = self.pooling(outputs_1)  # 26 / 10

        outputs_3 = self.conv_2(outputs_2)   # 24 /  8
        outputs_4 = self.pooling(outputs_3)  # 12 /  4

        outputs_5 = self.conv_3(outputs_4)   # 10 /  2
        outputs_flatten = self.flatten(outputs_5)

        # fully connected part
        dense = self.dense_0(outputs_flatten)  # 512
        dense = self.dropout(dense)
        dense = self.dense_1(dense)  # 128
        dense = self.dropout(dense)
        final_output = self.final_dense(dense)  # 1

        return final_output