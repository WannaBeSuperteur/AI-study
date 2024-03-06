import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from train import create_train_and_valid_data


class Classify_Nums_CNN_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[28, 28, 1])
        self.conv_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')

        self.conv_2 = tf.keras.layers.Conv2D(96, (3, 3), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(192, (3, 3), activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')

        # fully connected part
        self.dense = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                           kernel_regularizer=L2, name='dense_0')

        self.final_dense = tf.keras.layers.Dense(256, activation='softmax',
                                                 kernel_regularizer=L2, name='dense_final')


    def call(self, inputs, training):
        inputs = tf.keras.layers.reshape((28, 28, 1))

        # conv + pooling part : 28 -> 26 -> 24 -> 12 -> 10 -> 8 -> 6 -> 4
        outputs_0 = self.conv_0(inputs)
        outputs_1 = self.conv_1(outputs_0)
        outputs_2 = self.pooling(output_1)

        outputs_3 = self.conv_2(output_2)
        outputs_4 = self.conv_3(output_3)
        outputs_5 = self.conv_4(output_4)
        outputs_6 = self.conv_5(output_5)

        outputs_flatten = self.flatten(outputs_6)

        # fully connected part
        dense_0 = self.dense(outputs_flatten)
        dense_0 = self.dropout(dense_0)
        final_output = self.final_dense(dense_0)

        return final_output
    

# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = MainModel()
    return model, optimizer, early_stopping, lr_reduced


# 모델 학습 및 저장
def train_model(input_data_all, output_data_all):
    (train_input, train_output, valid_input, valid_output) = define_data(input_data_all, output_data_all)

    print(f'train input : {np.shape(train_input)}\n{train_input}\n')
    print(f'valid input : {np.shape(valid_input)}\n{valid_input}\n')
    print(f'train output : {np.shape(train_output)}\n{train_output}\n')
    print(f'valid output : {np.shape(valid_output)}\n{valid_output}\n')
    
    model, optimizer, early_stopping, lr_reduced = define_model()
    
    model.compile(loss='mse', optimizer=optimizer)

    model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=50,
        validation_data=(valid_input, valid_output)
    )

    model.summary()
    model.save('main_model')

    return model


# CNN 모델 학습
def train_cnn_model(train_input, train_class):
    cnn_model, optimizer, early_stopping, lr_reduced = define_model()
    cnn_model.compile(loss='mse', optimizer=optimizer)

    # TODO : 모델 학습 및 반환


if __name__ == '__main__':

    # 학습 데이터 받아오기
    train_input, train_class, _ = create_train_and_valid_data()    

    # CNN 모델 학습
    cnn_model = train_cnn_model()

    # CNN 모델 저장
    cnn_model.save('classify_nums_model')

