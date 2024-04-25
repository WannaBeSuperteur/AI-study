import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from keras.losses import mean_squared_error
import keras.backend as K
import matplotlib.pyplot as plt

import cv2


INPUT_IMG_SIZE = 120
NUM_CHANNELS = 3 # R, G, and B
TOTAL_CELLS = INPUT_IMG_SIZE * INPUT_IMG_SIZE
TOTAL_INPUT_IMG_VALUES = NUM_CHANNELS * TOTAL_CELLS
NUM_INFO = 11 # male prob, female prob, hair color, inv hair color, mouth, eyes, and face location from top/left/right, background mean, background std

BATCH_SIZE = 32
HIDDEN_DIMS = 231

MSE_LOSS_WEIGHT = 200000.0
TRAIN_EPOCHS = 150
TRAIN_DATA_LIMIT = None
SILU_MULTIPLE = 2.0 # same as GeLU approximation

print(f'settings: HIDDEN_DIMS={HIDDEN_DIMS}, MSE_LOSS_WEIGHT={MSE_LOSS_WEIGHT}, SILU_MULTIPLE={SILU_MULTIPLE}')


# 사용자 정의 activation function = x * sigmoid(1.702x)
def silu_mul(x):
    return x * K.sigmoid(SILU_MULTIPLE * x)


# random normal noise maker for VAE
def noise_maker(noise_args):
    noise_mean = noise_args[0]
    noise_log_var = noise_args[1]

    noise = K.random_normal(shape=(BATCH_SIZE, HIDDEN_DIMS), mean=0.0, stddev=1.0)
    return K.exp(noise_log_var / 2.0) * noise + noise_mean


# ref-1: https://www.kaggle.com/code/mersico/cvae-from-scratch
# ref-2: https://github.com/ekzhang/vae-cnn-mnist/blob/master/MNIST%20Convolutional%20VAE%20with%20Label%20Input.ipynb
class CVAE_Model:

    def get_mse_and_kl_loss(self, x, y):
        x_reshaped = K.reshape(x, shape=(BATCH_SIZE, TOTAL_INPUT_IMG_VALUES))
        y_reshaped = K.reshape(y, shape=(BATCH_SIZE, TOTAL_INPUT_IMG_VALUES))

        mse_loss = MSE_LOSS_WEIGHT * mean_squared_error(x_reshaped, y_reshaped)
        kl_loss = -0.5 * K.sum(1 + self.latent_log_var - K.square(self.latent_mean) - K.exp(self.latent_log_var), axis=-1)

#        self.mse_loss_tracker.update_state(mse_loss)
#        self.kl_loss_tracker.update_state(kl_loss)
#        self.total_loss_tracker.update_state(mse_loss + kl_loss)

        return mse_loss, kl_loss, y_reshaped


    # VAE 의 loss function
    def vae_loss(self, x, y):
        mse_loss, kl_loss, _ = self.get_mse_and_kl_loss(x, y)
        return mse_loss + kl_loss


    def __init__(self, dropout_rate=0.25):

        # loss tracker
        self.mse_loss_tracker = tf.keras.metrics.Mean(name='cvae_mse_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='cvae_kl_loss')
        self.total_loss_tracker = tf.keras.metrics.Mean(name='cvae_total_loss')

        # 공통 레이어
        self.flatten = tf.keras.layers.Flatten()
        self.flatten_for_ad0 = tf.keras.layers.Flatten()
        self.flatten_for_ad1 = tf.keras.layers.Flatten()
        self.flatten_for_ad2 = tf.keras.layers.Flatten()

        self.dropout_enc_c0 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_enc_c0')
        self.dropout_enc_c1 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_enc_c1')
        self.dropout_enc_c2 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_enc_c2')

        self.dropout_dec_c0 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_dec_c0')
        self.dropout_dec_c1 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_dec_c1')
        self.dropout_dec_c2 = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout_dec_c2')

        L2 = tf.keras.regularizers.l2(0.001)

        # encoder 용 레이어
        self.encoder_cnn0 = layers.Conv2D(32, (4, 4), strides=2, activation=silu_mul, padding='same', kernel_regularizer=L2, name='ec0')
        self.encoder_bn_cnn0 = layers.BatchNormalization(name='ec0_bn')
        self.encoder_ac_cnn0 = layers.Activation(silu_mul, name='ec0_ac')

        self.encoder_cnn1 = layers.Conv2D(48, (4, 4), strides=2, activation=silu_mul, padding='same', kernel_regularizer=L2, name='ec1')
        self.encoder_bn_cnn1 = layers.BatchNormalization(name='ec1_bn')
        self.encoder_ac_cnn1 = layers.Activation(silu_mul, name='ec1_ac')

        self.encoder_cnn2 = layers.Conv2D(96, (4, 4), strides=2, activation=silu_mul, padding='same', kernel_regularizer=L2, name='ec2')
        self.encoder_bn_cnn2 = layers.BatchNormalization(name='ec2_bn')
        self.encoder_ac_cnn2 = layers.Activation(silu_mul, name='ec2_ac')

        self.encoder_cnn3 = layers.Conv2D(192, (4, 4), strides=1, activation=silu_mul, padding='same', kernel_regularizer=L2, name='ec3')
        self.encoder_bn_cnn3 = layers.BatchNormalization(name='ec3_bn')
        self.encoder_ac_cnn3 = layers.Activation(silu_mul, name='ec3_ac')

        self.encoder_dense0 = layers.Dense(256, activation=silu_mul, name='ed0')
        self.encoder_ad0 = layers.Dense(64, activation=silu_mul, name='ead0') # input image 와 직접 연결
        self.encoder_ad1 = layers.Dense(64, activation=silu_mul, name='ead1') # 60 x 60 feature 와 직접 연결
        self.encoder_ad2 = layers.Dense(64, activation=silu_mul, name='ead2') # 30 x 30 feature 와 직접 연결

        # decoder 용 레이어
        self.decoder_dense0 = layers.Dense(256, activation=silu_mul, name='dd0')
        self.decoder_dense1 = layers.Dense(240 * TOTAL_CELLS // (8 * 8), activation=silu_mul, name='dd1')

        self.decoder_cnn0 = layers.Conv2DTranspose(120, (4, 4), strides=2, activation=silu_mul, padding='same', kernel_regularizer=L2, name='dc0')
        self.decoder_bn_cnn0 = layers.BatchNormalization(name='dc0_bn')
        self.decoder_ac_cnn0 = layers.Activation(silu_mul, name='dc0_ac')

        self.decoder_cnn1 = layers.Conv2DTranspose(60, (4, 4), strides=2, activation=silu_mul, padding='same', kernel_regularizer=L2, name='dc1')
        self.decoder_bn_cnn1 = layers.BatchNormalization(name='dc1_bn')
        self.decoder_ac_cnn1 = layers.Activation(silu_mul, name='dc1_ac')

        self.decoder_cnn2 = layers.Conv2DTranspose(40, (4, 4), strides=2, activation=silu_mul, padding='same', kernel_regularizer=L2, name='dc2')
        self.decoder_bn_cnn2 = layers.BatchNormalization(name='dc2_bn')
        self.decoder_ac_cnn2 = layers.Activation(silu_mul, name='dc2_ac')

        self.decoder_cnn3 = layers.Conv2D(NUM_CHANNELS, (4, 4), strides=1, activation='sigmoid', padding='same', kernel_regularizer=L2, name='dc3')
        self.decoder_bn_cnn3 = layers.BatchNormalization(name='dc3_bn')
        self.decoder_ac_cnn3 = layers.Activation(silu_mul, name='dc3_ac')

        # encoder (main stream)
        input_image = layers.Input(batch_shape=(BATCH_SIZE, INPUT_IMG_SIZE, INPUT_IMG_SIZE, NUM_CHANNELS), name='ec_input_img')
#        input_image_reshaped = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE, NUM_CHANNELS))(input_image)

        input_condition = layers.Input(shape=(NUM_INFO,), name='ec_input_cond')

        enc_c0 = self.encoder_cnn0(input_image) # input_image_reshaped
        enc_c0 = self.encoder_bn_cnn0(enc_c0)
        enc_c0 = self.encoder_ac_cnn0(enc_c0)
        enc_c0 = self.dropout_enc_c0(enc_c0)

        enc_c1 = self.encoder_cnn1(enc_c0)
        enc_c1 = self.encoder_bn_cnn1(enc_c1)
        enc_c1 = self.encoder_ac_cnn1(enc_c1)
        enc_c1 = self.dropout_enc_c1(enc_c1)

        enc_c2 = self.encoder_cnn2(enc_c1)
        enc_c2 = self.encoder_bn_cnn2(enc_c2)
        enc_c2 = self.encoder_ac_cnn2(enc_c2)
        enc_c2 = self.dropout_enc_c2(enc_c2)

        enc_c3 = self.encoder_cnn3(enc_c2)
        enc_c3 = self.encoder_bn_cnn3(enc_c3)
        enc_c3 = self.encoder_ac_cnn3(enc_c3)
        enc_flatten = self.flatten(enc_c3)

        enc_merged = layers.concatenate([enc_flatten, input_condition])
        enc_d0 = self.encoder_dense0(enc_merged)

        # encoder (additional stream)
        # original 120 x 120 image
        enc_flatten_for_ad0 = self.flatten_for_ad0(input_image)
        enc_flatten_for_ad0 = layers.concatenate([enc_flatten_for_ad0, input_condition])
        enc_ad0 = self.encoder_ad0(enc_flatten_for_ad0)

        # 60 x 60 feature map
        enc_flatten_for_ad1 = self.flatten_for_ad1(enc_c0)
        enc_flatten_for_ad1 = layers.concatenate([enc_flatten_for_ad1, input_condition])
        enc_ad1 = self.encoder_ad1(enc_flatten_for_ad1)

        # 30 x 30 feature map
        enc_flatten_for_ad2 = self.flatten_for_ad2(enc_c1)
        enc_flatten_for_ad2 = layers.concatenate([enc_flatten_for_ad2, input_condition])
        enc_ad2 = self.encoder_ad2(enc_flatten_for_ad2)

        # encoder (concatenated)
        enc_d0_ad = layers.concatenate([enc_d0, enc_ad0, enc_ad1, enc_ad2, input_condition])

        # latent space
        self.latent_mean = layers.Dense(HIDDEN_DIMS, name='lm', activation='tanh')(enc_d0_ad)
        self.latent_log_var = layers.Dense(HIDDEN_DIMS, name='llv', activation='tanh')(enc_d0_ad)
        self.latent_space = layers.Lambda(noise_maker, output_shape=(HIDDEN_DIMS,), name='ls')([self.latent_mean, self.latent_log_var])

        # decoder
        latent_for_decoder = layers.Input(batch_shape=(BATCH_SIZE, HIDDEN_DIMS), name='dc_input_latent') # shape=(HIDDEN_DIMS,)
        condition_for_decoder = layers.Input(batch_shape=(BATCH_SIZE, NUM_INFO), name='dc_input_cond') # shape=(NUM_INFO,)

        dec_merged = layers.concatenate([latent_for_decoder, condition_for_decoder])
        dec_d0 = self.decoder_dense0(dec_merged)
        dec_d1 = self.decoder_dense1(dec_d0)
        dec_reshaped = layers.Reshape((INPUT_IMG_SIZE // 8, INPUT_IMG_SIZE // 8, 240))(dec_d1)

        # decoder additional
        dec_add_c0 = layers.Dense((INPUT_IMG_SIZE // 8) * (INPUT_IMG_SIZE // 8) * 240, name='dec_c0_add', activation=silu_mul)(dec_merged)
        dec_add_c0_ = layers.Reshape((INPUT_IMG_SIZE // 8, INPUT_IMG_SIZE // 8, 240))(dec_add_c0)

        dec_add_c1 = layers.Dense((INPUT_IMG_SIZE // 4) * (INPUT_IMG_SIZE // 4) * 120, name='dec_c1_add', activation=silu_mul)(dec_merged)
        dec_add_c1_ = layers.Reshape((INPUT_IMG_SIZE // 4, INPUT_IMG_SIZE // 4, 120))(dec_add_c1)

        dec_add_c2 = layers.Dense((INPUT_IMG_SIZE // 2) * (INPUT_IMG_SIZE // 2) * 60, name='dec_c2_add', activation=silu_mul)(dec_merged)
        dec_add_c2_ = layers.Reshape((INPUT_IMG_SIZE // 2, INPUT_IMG_SIZE // 2, 60))(dec_add_c2)

        dec_add_c3 = layers.Dense(INPUT_IMG_SIZE * INPUT_IMG_SIZE * 40, name='dec_c3_add', activation=silu_mul)(dec_merged)
        dec_add_c3_ = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE, 40))(dec_add_c3)

        # decoder deconv CNN layers
        dec_c0 = self.decoder_cnn0(layers.Concatenate()([dec_reshaped, dec_add_c0_]))
        dec_c0 = self.decoder_bn_cnn0(dec_c0)
        dec_c0 = self.decoder_ac_cnn0(dec_c0)
        dec_c0 = self.dropout_dec_c0(dec_c0)

        dec_c1 = self.decoder_cnn1(layers.Concatenate()([dec_c0, dec_add_c1_]))
        dec_c1 = self.decoder_bn_cnn1(dec_c1)
        dec_c1 = self.decoder_ac_cnn1(dec_c1)
        dec_c1 = self.dropout_dec_c1(dec_c1)

        dec_c2 = self.decoder_cnn2(layers.Concatenate()([dec_c1, dec_add_c2_]))
        dec_c2 = self.decoder_bn_cnn2(dec_c2)
        dec_c2 = self.decoder_ac_cnn2(dec_c2)
        dec_c2 = self.dropout_dec_c2(dec_c2)

        dec_c3 = self.decoder_cnn3(layers.Concatenate()([dec_c2, dec_add_c3_]))
        dec_c3 = self.decoder_bn_cnn3(dec_c3)
        dec_c3 = self.decoder_ac_cnn3(dec_c3)

        dec_final = layers.Reshape((INPUT_IMG_SIZE, INPUT_IMG_SIZE, NUM_CHANNELS))(dec_c3)

        # define encoder, decoder and cvae model
        self.encoder = tf.keras.Model([input_image, input_condition], self.latent_space, name='encoder')
        self.decoder = tf.keras.Model([latent_for_decoder, condition_for_decoder], dec_final, name='decoder')

        self.cvae = tf.keras.Model(
            inputs=[input_image, input_condition, condition_for_decoder],
            outputs=self.decoder([self.encoder([input_image, input_condition]), condition_for_decoder]),
            name='final_cvae'
        )

    """
        # to solve error below:
        # INVALID_ARGUMENT: You must feed a value for placeholder tensor 'input_3' with dtype float and shape [32,76]
        self.feed_to_solve_invalid_arg(
            input_img=input_image,
            input_cond=input_condition,
            dec_cond=condition_for_decoder,
            dec_output=dec_final
        )
        self.feed_to_solve_invalid_arg_for_decoder(
            latent_vector=latent_for_decoder,
            dec_cond=condition_for_decoder,
            dec_output=dec_final
        )

    def feed_to_solve_invalid_arg(self, input_img, input_cond, dec_cond, dec_output):
        with tf.compat.v1.Session() as sess:
            image = np.random.normal(size=(BATCH_SIZE, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3))
            cond = np.random.uniform(size=(BATCH_SIZE, NUM_INFO))

            sess.run(dec_output, feed_dict={input_img: image, input_cond: cond, dec_cond: cond})


    def feed_to_solve_invalid_arg_for_decoder(self, latent_vector, dec_cond, dec_output):
        with tf.compat.v1.Session() as sess:
            latent = np.random.normal(size=(BATCH_SIZE, HIDDEN_DIMS))
            cond = np.random.uniform(size=(BATCH_SIZE, NUM_INFO))

            sess.run(dec_output, feed_dict={latent_vector: latent, dec_cond: cond})
    """

    def call(self, inputs, training):
        return self.cvae(inputs)


# C-VAE 모델의 learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 4:
        return lr
    elif lr > 0.0003:
        return lr * 0.9675
    elif lr > 0.0001:
        return lr * 0.9875
    else:
        return lr


# C-VAE 모델 정의 및 반환
def define_cvae_model():
    optimizer = optimizers.Adam(0.0006, decay=1e-6)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model = CVAE_Model(dropout_rate=0.25) # 실제 모델은 model.cvae

    return model, optimizer, scheduler_callback


# model architecture image
def plot_model_architecture(model, img_file_name):
    try:
        tf.keras.utils.plot_model(model, to_file=f'{img_file_name}.png', show_shapes=True)
    except Exception as e:
        print(f'model architecture image file generation error : {e}')


# 모델 구조 표시
def show_model_summary(model_class):
    print('\n === ENCODER ===')
    model_class.encoder.summary()
    plot_model_architecture(model_class.encoder, 'encoder')

    print('\n === DECODER ===')
    model_class.decoder.summary()
    plot_model_architecture(model_class.decoder, 'decoder')

    print('\n === VAE ===')
    model_class.cvae.summary()
    plot_model_architecture(model_class.cvae, 'cvae')


class LossDetailCallback(tf.keras.callbacks.Callback):
    def __init__(self, cvae_model_class: CVAE_Model):
        super().__init__()
        self.cvae_class = cvae_model_class


    def on_epoch_end(self, epoch, logs=None):
        mse_loss = tf.print(self.cvae_class.mse_loss_tracker.result())
        kl_loss = tf.print(self.cvae_class.kl_loss_tracker.result())
        total_loss = tf.print(self.cvae_class.total_loss_tracker.result())

#        loss_dict = {'mse_loss': mse_loss, 'kl_loss': kl_loss, 'total_loss': total_loss}
        print(f' - mse_loss: {mse_loss} - kl_loss: {kl_loss} - total_loss: {total_loss}')


# C-VAE 모델 학습 loss 기록 저장
def save_cvae_loss_log(train_history):
    plt.plot(train_history.history['loss'])
    plt.title('CVAE loss history')
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.savefig('cvae_train_result.png')


# C-VAE 모델 학습 실시 및 모델 저장
# train_info = train_condition (N, 5)
def train_cvae_model(train_input, train_info):

    # to solve "You must feed a value for placeholder tensor {tensor_name} with dtype float and shape {shape}."
    tf.keras.backend.set_learning_phase(False)
    cvae_model_class, optimizer, scheduler_callback = define_cvae_model()
    cvae_model_class.cvae.compile(loss=cvae_model_class.vae_loss, optimizer=optimizer)

    # 학습 실시
    train_history = cvae_model_class.cvae.fit(
        [train_input, train_info, train_info], train_input,
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
#        callbacks=[scheduler_callback, LossDetailCallback(cvae_model_class=cvae_model_class)],
        callbacks=[scheduler_callback],
        shuffle=True
    )
    save_cvae_loss_log(train_history)

    # 모델 구조 표시
    show_model_summary(cvae_model_class)

    cvae_model_class.encoder.save('cvae_encoder_model')
    cvae_model_class.decoder.save('cvae_decoder_model')
    cvae_model_class.cvae.save('cvae_model')

    return cvae_model_class.encoder, cvae_model_class.decoder, cvae_model_class.cvae


# 학습 데이터에 필요한 자료 (이미지 및 male/female prob, hair color, mouth, eyes info) 를 가져와서 학습 데이터 생성
def create_train_and_valid_data(limit=None):
    condition_data = pd.read_csv('condition_data.csv', index_col=0)
    print(condition_data)

    train_input = []
    train_info = []
    current_idx = 0

    for _, row in condition_data.iterrows():
        if current_idx % 250 == 0:
            print(current_idx)

        img_path = row['image_path']

        # condition info (total 11)
        male_prob = row['male_prob']
        female_prob = row['female_prob']
        hair_color = row['hair_color']
        inv_hair_color = 1.0 - row['hair_color']
        mouth = row['mouth']
        eyes = row['eyes']
        face_location_top = row['face_location_top_normalized']
        face_location_left = row['face_location_left_normalized']
        face_location_right = row['face_location_right_normalized']
        background_mean = row['background_mean']
        background_std = row['background_std']

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        train_input.append(np.array(img) / 255.0)
        train_info.append([male_prob, female_prob, hair_color, inv_hair_color, mouth, eyes,
                           face_location_top, face_location_left, face_location_right,
                           background_mean, background_std])

        current_idx += 1
        if limit is not None and current_idx >= limit:
            break

    # batch size is 32 -> only "multiple of 32" is available total dataset size
    available_length = len(train_input) // BATCH_SIZE * BATCH_SIZE
    print(f'total dataset size : {available_length}')

    train_input = np.array(train_input)[:available_length]
    train_info = np.array(train_info)[:available_length]

    return train_input, train_info


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    np.set_printoptions(suppress=True, linewidth=160)
    pd.set_option('display.max_columns', 16)

    # 학습 데이터 추출 (이미지 input + 해당 이미지의 class)
    train_input, train_info = create_train_and_valid_data(limit=TRAIN_DATA_LIMIT)
    
    print(f'\nshape of train input: {np.shape(train_input)}')
    print(train_input)
    
    print(f'\nshape of train info: {np.shape(train_info)}')
    print(train_info)

    # 학습 실시 및 모델 저장
    cvae_encoder, cvae_decoder, cvae_model = train_cvae_model(train_input, train_info)

