# from GAI-P2

import tensorflow as tf

IMG_WIDTH = 104
IMG_HEIGHT = 128


class Regression_Head_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # height : 128 -> 126 -> 124 -> 62 -> 60 -> 30 -> 28 -> 14 -> 12
        # width  : 104 -> 102 -> 100 -> 50 -> 48 -> 24 -> 22 -> 11 ->  9
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
        # height : 128 -> 126 -> 124 -> 62 -> 60 -> 30 -> 28 -> 14 -> 12
        # width  : 104 -> 102 -> 100 -> 50 -> 48 -> 24 -> 22 -> 11 ->  9
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
        dense = self.dense_0(outputs_flatten)  # 256
        dense = self.dropout(dense)
        dense = self.dense_1(dense)  # 64
        dense = self.dropout(dense)
        final_output = self.final_dense(dense)  # 1

        return final_output