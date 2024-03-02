import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import binary_crossentropy
import keras.backend as K

INPUT_IMG_SIZE = 28
NUM_CLASSES = 10
HIDDEN_DIMS = 16
TOTAL_CELLS = INPUT_IMG_SIZE * INPUT_IMG_SIZE


# CVAE Model
# 입력 이미지 (-1, 28, 28), 입력 class (10,), 출력 이미지 (-1, 28, 28)

# ref: https://www.kaggle.com/code/mersico/cvae-from-scratch
class CVAE_Model:

    def noise_maker(self, noise_args):
        noise_mean = noise_args[0]
        noise_log_var = noise_args[1]
        
        noise = K.random_normal(shape=(-1, HIDDEN_DIMS), mean=0.0, stddev=1.0)
        return K.exp(self.latent_log_var / 2.0) * noise + self.latent_mean


    # VAE 의 loss function (kl_loss : KL Divergence, https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
    def vae_loss(self, x, y):
        x_reshaped = np.reshape(x, shape=(-1, TOTAL_CELLS))
        y_reshaped = np.reshape(y, shape=(-1, TOTAL_CELLS))
        print(np.shape(x_reshaped))
        bce_loss = TOTAL_CELLS * binary_crossentropy(x_reshaped, y_reshaped)
        
        kl_loss = -0.5 * K.sum(1 + self.latent_log_var - K.square(self.latent_mean) - K.exp(self.latent_log_var), axis=-1)
        print(kl_loss)
        print(np.shape(kl_loss))
        return mse_loss + kl_loss
    

    def __init__(self, dropout_rate=0.45):

        # 공통 레이어
        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.unpooling = tfa.layers.MaxUnpooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')

        L2 = tf.keras.regularizers.l2(0.001)

        # encoder 용 레이어
        self.encoder_cnn0 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='ec0')
        self.encoder_cnn1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='ec1')
        self.encoder_cnn2 = layers.Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='ec2')
        self.encoder_cnn3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='ec3')

        self.encoder_dense0 = layers.Dense(256, activation='relu', name='ed0')
        self.encoder_dense1 = layers.Dense(64, activation='relu', name='ed1')

        # decoder 용 레이어
        self.decoder_dense0 = layers.Dense(64, activation='relu', name='dd0')
        self.decoder_dense1 = layers.Dense(256, activation='relu', name='dd1')
        self.decoder_dense2 = layers.Dense(128 * TOTAL_CELLS, activation='relu', name='dd2')

        self.decoder_cnn0 = layers.Conv2D(96, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='dc0')
        self.decoder_cnn1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='dc1')
        self.decoder_cnn2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='dc2')
        self.decoder_cnn3 = layers.Conv2D(1, (3, 3), activation='relu', padding='same', kernel_regularizer=L2, name='dc3')

        # encoder
        input_image = layers.Input(batch_shape=(None, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 1))
        input_class = layers.Input(shape=(NUM_CLASSES,))
        
        enc_c0 = self.encoder_cnn0(input_image)
        enc_c0 = self.dropout(enc_c0)
        enc_c1 = self.encoder_cnn1(enc_c0)
        enc_c1 = self.pooling(enc_c1)
        enc_c1 = self.dropout(enc_c1)
        enc_c2 = self.encoder_cnn2(enc_c1)
        enc_c2 = self.pooling(enc_c2)
        enc_c2 = self.dropout(enc_c2)
        enc_c3 = self.encoder_cnn3(enc_c2)
        enc_fl = self.flatten(enc_c3)

        enc_merged = layers.concatenate([enc_fl, input_class])
        enc_d0 = self.encoder_dense0(enc_merged)
        enc_d0 = self.dropout(enc_d0)
        enc_d1 = self.encoder_dense1(enc_d0)

        # latent space
        self.latent_mean = layers.Dense(HIDDEN_DIMS, name='lm')(enc_d1)
        self.latent_log_var = layers.Dense(HIDDEN_DIMS, name='llv')(enc_d1)
        self.latent_space = layers.Lambda(self.noise_maker, output_shape=(HIDDEN_DIMS,), name='ls')([self.latent_mean, self.latent_log_var])

        # decoder
        latent_for_decoder = Input(shape=(HIDDEN_DIMS,))
        class_for_decoder = Input(shape=(NUM_CLASSES,))

        dec_merged = layers.concatenate([latent_for_decoder, class_for_decoder])
        dec_d0 = self.decoder_dense0(dec_merged)
        dec_d0 = self.dropout(dec_d0)
        dec_d1 = self.decoder_dense1(dec_d0)
        dec_d2 = self.decoder_dense2(dec_d1)
        dec_reshaped = layers.Reshape((INPUT_IMG_SIZE // 4, INPUT_IMG_SIZE // 4, 1))(dec_c2)

        dec_c0 = self.decoder_cnn0(dec_reshaped)
        dec_c0 = self.unpooling(dec_c0)
        dec_c0 = self.dropout(dec_c0)
        dec_c1 = self.decoder_cnn1(dec_c0)
        dec_c1 = self.unpooling(dec_c1)
        dec_c1 = self.dropout(dec_c1)
        dec_c2 = self.decoder_cnn2(dec_c1)
        dec_c2 = self.dropout(dec_c2)
        dec_c3 = self.decoder_cnn3(dec_c2)
        dec_final = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE))(dec_c3)

        # define encoder, decoder and cvae model
        self.encoder = keras.Model([input_image, input_class], self.latent_space, name='encoder')
        self.decoder = keras.Model([latent_for_decoder, class_for_decoder], dec_final, name='decoder')

        self.cvae = keras.Model(
            inputs=[input_image, input_class, class_for_decoder],
            outputs=self.decoder([self.encoder([input_image, input_class]), class_for_decoder]),
            name='final_cvae'
        )


#    def call(self, inputs, training):
#        return self.cvae(inputs)


# mnist_train.csv 파일로부터 학습 데이터 추출
def create_train_and_valid_data():
    train_df = pd.read_csv('mnist_train.csv')
    train_n = len(train_df)
    print(train_df)

    train_input = np.zeros((train_n, INPUT_IMG_SIZE, INPUT_IMG_SIZE))
    train_class = np.zeros((train_n, NUM_CLASSES))

    for idx, row in train_df.iterrows():
        if idx % 5000 == 0:
            print(idx)

        # input
        inp = np.array(row['1x1':'28x28'].to_list())
        inp = inp.reshape((28, 28))
        inp = inp / 255.0
        train_input[idx] = inp

        # output
        out_class = int(row['label'])
        train_class[idx][out_class] = 1

    return train_input, train_class


# C-VAE 모델 정의 및 반환
def define_cvae_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    model = CVAE_Model(dropout_rate=0.45) # 실제 모델은 model.cvae
    return model, optimizer


# C-VAE 모델 학습 실시 및 모델 저장
def train_cvae_model(train_input, train_class):
    cvae_model, optimizer = define_cvae_model()
    cvae_model.compile(loss=cvae_model.vae_loss, optimizer=optimizer)

    cvae_model.cvae.fit(
        [train_input, train_class, train_class], train_input,
        epochs=30,
        shuffle=True
    )

    cvae_model.cvae.summary()
    cvae_model.cvae.save('cvae_model')
    return cvae_model


if __name__ == '__main__':

    # 학습 데이터 추출 (이미지 input + 해당 이미지의 class)
    train_input, train_class = create_train_and_valid_data()
    print(f'shape of train input: {np.shape(train_input)}')
    print(f'shape of train class: {np.shape(train_class)}')

    # 학습 실시 및 모델 저장
    cvae_model = train_cvae_model(train_input, train_class)
