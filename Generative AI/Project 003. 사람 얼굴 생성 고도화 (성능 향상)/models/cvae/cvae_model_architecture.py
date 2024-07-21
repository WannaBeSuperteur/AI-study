# from GAI-P2

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from keras.losses import mean_squared_error
import keras.backend as K
from cvae_model_utils import scheduler, save_cvae_loss_log, show_model_summary


# 사용자 정의 activation function = x * sigmoid(2x) (x * sigmoid(1.702x) in GeLU approximation)
INPUT_IMG_WIDTH = 104
INPUT_IMG_HEIGHT = 128
NUM_CHANNELS = 3 # R, G, and B
TOTAL_CELLS = INPUT_IMG_WIDTH * INPUT_IMG_HEIGHT
TOTAL_INPUT_IMG_VALUES = NUM_CHANNELS * TOTAL_CELLS
NUM_INFO = 5  # hair_color, mouth, eyes, head, background

BATCH_SIZE = 32
HIDDEN_DIMS = 120

MSE_LOSS_WEIGHT = 50000.0
TRAIN_EPOCHS = 1  # 60
TRAIN_DATA_LIMIT = None
SILU_MULTIPLE = 2.0

print(f'settings: HIDDEN_DIMS={HIDDEN_DIMS}, MSE_LOSS_WEIGHT={MSE_LOSS_WEIGHT}, SILU_MULTIPLE={SILU_MULTIPLE}')


def silu_mul(x):
    return x * K.sigmoid(SILU_MULTIPLE * x)


# random normal noise maker for VAE
def noise_maker(noise_args):
    noise_mean = noise_args[0]
    noise_log_var = noise_args[1]

    noise = K.random_normal(shape=(BATCH_SIZE, HIDDEN_DIMS), mean=0.0, stddev=1.0)
    return K.exp(noise_log_var / 2.0) * noise + noise_mean


# TODO complete model architecture

class CVAE_Model:

    def get_mse_and_kl_loss(self, x, y):
        x_reshaped = K.reshape(x, shape=(BATCH_SIZE, TOTAL_INPUT_IMG_VALUES))
        y_reshaped = K.reshape(y, shape=(BATCH_SIZE, TOTAL_INPUT_IMG_VALUES))

        mse_loss = MSE_LOSS_WEIGHT * mean_squared_error(x_reshaped, y_reshaped)
        kl_loss0 = -0.5 * K.sum(1 + self.latent0_log_var - K.square(self.latent0_mean) - K.exp(self.latent0_log_var), axis=-1)
        kl_loss1 = -0.5 * K.sum(1 + self.latent1_log_var - K.square(self.latent1_mean) - K.exp(self.latent1_log_var), axis=-1)
        kl_loss2 = -0.5 * K.sum(1 + self.latent2_log_var - K.square(self.latent2_mean) - K.exp(self.latent2_log_var), axis=-1)
        kl_loss3 = -0.5 * K.sum(1 + self.latent3_log_var - K.square(self.latent3_mean) - K.exp(self.latent3_log_var), axis=-1)
        kl_loss = kl_loss0 + kl_loss1 + kl_loss2 + kl_loss3

        return mse_loss, kl_loss, y_reshaped


    # VAE 의 loss function
    def vae_loss(self, x, y):
        mse_loss, kl_loss, _ = self.get_mse_and_kl_loss(x, y)
        return mse_loss + kl_loss

    def define_common_layers(self):
        """
        define common layers for CVAE model
        """

        self.flatten = tf.keras.layers.Flatten()
        self.flatten_for_ad0 = tf.keras.layers.Flatten()
        self.flatten_for_ad1 = tf.keras.layers.Flatten()
        self.flatten_for_ad2 = tf.keras.layers.Flatten()

        self.dropout_enc_c0 = tf.keras.layers.Dropout(rate=self.dropout_rate, name='dropout_enc_c0')
        self.dropout_enc_c1 = tf.keras.layers.Dropout(rate=self.dropout_rate, name='dropout_enc_c1')
        self.dropout_enc_c2 = tf.keras.layers.Dropout(rate=self.dropout_rate, name='dropout_enc_c2')

        self.dropout_dec_c0 = tf.keras.layers.Dropout(rate=self.dropout_rate, name='dropout_dec_c0')
        self.dropout_dec_c1 = tf.keras.layers.Dropout(rate=self.dropout_rate, name='dropout_dec_c1')
        self.dropout_dec_c2 = tf.keras.layers.Dropout(rate=self.dropout_rate, name='dropout_dec_c2')

        self.L2 = tf.keras.regularizers.l2(0.001)

    def define_encoder_layers(self):
        """
        define layers for CVAE encoder
        """

        self.encoder_cnn0 = layers.Conv2D(32, (3, 3), strides=2, activation=silu_mul,
                                          padding='same', kernel_regularizer=self.L2, name='ec0')
        self.encoder_bn_cnn0 = layers.BatchNormalization(name='ec0_bn')
        self.encoder_ac_cnn0 = layers.Activation(silu_mul, name='ec0_actv')

        self.encoder_cnn1 = layers.Conv2D(64, (3, 3), strides=2, activation=silu_mul,
                                          padding='same', kernel_regularizer=self.L2, name='ec1')
        self.encoder_bn_cnn1 = layers.BatchNormalization(name='ec1_bn')
        self.encoder_ac_cnn1 = layers.Activation(silu_mul, name='ec1_actv')

        self.encoder_cnn2 = layers.Conv2D(128, (3, 3), strides=2, activation=silu_mul,
                                          padding='same', kernel_regularizer=self.L2, name='ec2')
        self.encoder_bn_cnn2 = layers.BatchNormalization(name='ec2_bn')
        self.encoder_ac_cnn2 = layers.Activation(silu_mul, name='ec2_actv')

        self.encoder_cnn3 = layers.Conv2D(256, (3, 3), strides=1, activation=silu_mul,
                                          padding='same', kernel_regularizer=self.L2, name='ec3')
        self.encoder_bn_cnn3 = layers.BatchNormalization(name='ec3_bn')
        self.encoder_ac_cnn3 = layers.Activation(silu_mul, name='ec3_actv')

        self.encoder_cnn0_ad = layers.Dense(128, activation=silu_mul, name='ec0_ad')  # 52 x 64 feature 와 직접 연결
        self.encoder_cnn1_ad = layers.Dense(128, activation=silu_mul, name='ec1_ad')  # 52 x 64 feature 와 직접 연결
        self.encoder_cnn2_ad = layers.Dense(128, activation=silu_mul, name='ec2_ad')  # 26 x 32 feature 와 직접 연결
        self.encoder_cnn3_ad = layers.Dense(128, activation=silu_mul, name='ec3_ad')  # 13 x 16 feature 와 직접 연결


    def define_decoder_layers(self):
        """
        define layers for CVAE decoder
        """

        self.decoder_dense0 = layers.Dense(256, activation=silu_mul, name='dd0')
        self.decoder_dense1 = layers.Dense(256 * TOTAL_CELLS // (8 * 8), activation=silu_mul, name='dd1')

        self.decoder_cnn0 = layers.Conv2DTranspose(128, (3, 3),
                                                   strides=2, activation=silu_mul, padding='same',
                                                   kernel_regularizer=self.L2, name='dc0')
        self.decoder_bn_cnn0 = layers.BatchNormalization(name='dc0_bn')
        self.decoder_ac_cnn0 = layers.Activation(silu_mul, name='dc0_actv')

        self.decoder_cnn1 = layers.Conv2DTranspose(64, (3, 3),
                                                   strides=2, activation=silu_mul, padding='same',
                                                   kernel_regularizer=self.L2, name='dc1')
        self.decoder_bn_cnn1 = layers.BatchNormalization(name='dc1_bn')
        self.decoder_ac_cnn1 = layers.Activation(silu_mul, name='dc1_actv')

        self.decoder_cnn2 = layers.Conv2DTranspose(32, (3, 3),
                                                   strides=2, activation=silu_mul, padding='same',
                                                   kernel_regularizer=self.L2, name='dc2')
        self.decoder_bn_cnn2 = layers.BatchNormalization(name='dc2_bn')
        self.decoder_ac_cnn2 = layers.Activation(silu_mul, name='dc2_actv')

        self.decoder_cnn3 = layers.Conv2D(NUM_CHANNELS, (3, 3),
                                          strides=1, activation='sigmoid', padding='same',
                                          kernel_regularizer=self.L2, name='dc3')
        self.decoder_bn_cnn3 = layers.BatchNormalization(name='dc3_bn')
        self.decoder_ac_cnn3 = layers.Activation(silu_mul, name='dc3_actv')


    def define_conditional_value_layers(self):
        """
        define layers for conditional values attached to CVAE encoder/decoder
        """

        self.latent_for_decoder = layers.Input(batch_shape=(BATCH_SIZE, HIDDEN_DIMS), name='dc_input_latent')  # shape=(HIDDEN_DIMS,)
        self.condition_for_decoder = layers.Input(batch_shape=(BATCH_SIZE, NUM_INFO), name='dc_input_cond')  # shape=(NUM_INFO,)

        self.dec_merged = layers.concatenate([self.latent_for_decoder, self.condition_for_decoder])
        self.dec_d0 = self.decoder_dense0(self.dec_merged)
        self.dec_d1 = self.decoder_dense1(self.dec_d0)
        self.dec_reshaped = layers.Reshape((INPUT_IMG_HEIGHT // 8, INPUT_IMG_WIDTH // 8, 256))(self.dec_d1)

        self.dec_add_c0 = layers.Dense((INPUT_IMG_HEIGHT // 8) * (INPUT_IMG_WIDTH // 8) * 256, name='dec_c0_add', activation=silu_mul)(self.dec_merged)
        self.dec_add_c0_ = layers.Reshape((INPUT_IMG_HEIGHT // 8, INPUT_IMG_WIDTH // 8, 256))(self.dec_add_c0)

        self.dec_add_c1 = layers.Dense((INPUT_IMG_HEIGHT // 4) * (INPUT_IMG_WIDTH // 4) * 128, name='dec_c1_add', activation=silu_mul)(self.dec_merged)
        self.dec_add_c1_ = layers.Reshape((INPUT_IMG_HEIGHT // 4, INPUT_IMG_WIDTH // 4, 128))(self.dec_add_c1)

        self.dec_add_c2 = layers.Dense((INPUT_IMG_HEIGHT // 2) * (INPUT_IMG_WIDTH // 2) * 64, name='dec_c2_add', activation=silu_mul)(self.dec_merged)
        self.dec_add_c2_ = layers.Reshape((INPUT_IMG_HEIGHT // 2, INPUT_IMG_WIDTH // 2, 64))(self.dec_add_c2)

        self.dec_add_c3 = layers.Dense(INPUT_IMG_HEIGHT * INPUT_IMG_WIDTH * 32, name='dec_c3_add', activation=silu_mul)(self.dec_merged)
        self.dec_add_c3_ = layers.Reshape((INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 32))(self.dec_add_c3)


    def __init__(self, dropout_rate=0.25):

        # 1. 공통 레이어
        self.dropout_rate = dropout_rate
        self.define_common_layers()

        # 2. 개별 계층 레이어 정의
        # encoder 용 레이어, decoder 용 레이어, decoder additional (conditional value 용 레이어), latent vector 용 레이어
        self.define_encoder_layers()
        self.define_decoder_layers()
        self.define_conditional_value_layers()

        # 3. encoder (main stream)
        input_image = layers.Input(batch_shape=(BATCH_SIZE, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, NUM_CHANNELS), name='ec_input_img')
        input_condition = layers.Input(shape=(NUM_INFO,), name='ec_input_cond')

        enc_c0 = self.encoder_cnn0(input_image)  # input_image_reshaped
        enc_c0 = self.encoder_bn_cnn0(enc_c0)
        enc_c0 = self.encoder_ac_cnn0(enc_c0)
        enc_c0 = self.dropout_enc_c0(enc_c0)
        enc_c0_flatten = self.flatten(enc_c0)
        enc_c0_merged = layers.concatenate([enc_c0_flatten, input_condition])
        enc_c0_ad = self.encoder_cnn0_ad(enc_c0_merged)

        enc_c1 = self.encoder_cnn1(enc_c0)
        enc_c1 = self.encoder_bn_cnn1(enc_c1)
        enc_c1 = self.encoder_ac_cnn1(enc_c1)
        enc_c1 = self.dropout_enc_c1(enc_c1)
        enc_c1_flatten = self.flatten(enc_c1)
        enc_c1_merged = layers.concatenate([enc_c1_flatten, input_condition])
        enc_c1_ad = self.encoder_cnn1_ad(enc_c1_merged)

        enc_c2 = self.encoder_cnn2(enc_c1)
        enc_c2 = self.encoder_bn_cnn2(enc_c2)
        enc_c2 = self.encoder_ac_cnn2(enc_c2)
        enc_c2 = self.dropout_enc_c2(enc_c2)
        enc_c2_flatten = self.flatten(enc_c2)
        enc_c2_merged = layers.concatenate([enc_c2_flatten, input_condition])
        enc_c2_ad = self.encoder_cnn2_ad(enc_c2_merged)

        enc_c3 = self.encoder_cnn3(enc_c2)
        enc_c3 = self.encoder_bn_cnn3(enc_c3)
        enc_c3 = self.encoder_ac_cnn3(enc_c3)
        enc_c3_flatten = self.flatten(enc_c3)
        enc_c3_merged = layers.concatenate([enc_c3_flatten, input_condition])
        enc_c3_ad = self.encoder_cnn3_ad(enc_c3_merged)

        # 4. latent spaces
        self.latent0_mean = layers.Dense(HIDDEN_DIMS, name='lm0', activation='tanh')(enc_c0_ad)
        self.latent0_log_var = layers.Dense(HIDDEN_DIMS, name='llv0', activation='tanh')(enc_c0_ad)
        self.latent0_space = layers.Lambda(noise_maker, output_shape=(HIDDEN_DIMS,), name='ls0')([self.latent0_mean, self.latent0_log_var])

        self.latent1_mean = layers.Dense(HIDDEN_DIMS, name='lm1', activation='tanh')(enc_c1_ad)
        self.latent1_log_var = layers.Dense(HIDDEN_DIMS, name='llv1', activation='tanh')(enc_c1_ad)
        self.latent1_space = layers.Lambda(noise_maker, output_shape=(HIDDEN_DIMS,), name='ls1')([self.latent1_mean, self.latent1_log_var])

        self.latent2_mean = layers.Dense(HIDDEN_DIMS, name='lm2', activation='tanh')(enc_c2_ad)
        self.latent2_log_var = layers.Dense(HIDDEN_DIMS, name='llv2', activation='tanh')(enc_c2_ad)
        self.latent2_space = layers.Lambda(noise_maker, output_shape=(HIDDEN_DIMS,), name='ls2')([self.latent2_mean, self.latent2_log_var])

        self.latent3_mean = layers.Dense(HIDDEN_DIMS, name='lm3', activation='tanh')(enc_c3_ad)
        self.latent3_log_var = layers.Dense(HIDDEN_DIMS, name='llv3', activation='tanh')(enc_c3_ad)
        self.latent3_space = layers.Lambda(noise_maker, output_shape=(HIDDEN_DIMS,), name='ls3')([self.latent3_mean, self.latent3_log_var])

        # 5. concatenate latent values and conditions to create decoder input
        latent_for_decoder0 = layers.Input(batch_shape=(BATCH_SIZE, HIDDEN_DIMS), name='dc_input_latent_0')  # shape=(HIDDEN_DIMS,)
        latent_for_decoder1 = layers.Input(batch_shape=(BATCH_SIZE, HIDDEN_DIMS), name='dc_input_latent_1')  # shape=(HIDDEN_DIMS,)
        latent_for_decoder2 = layers.Input(batch_shape=(BATCH_SIZE, HIDDEN_DIMS), name='dc_input_latent_2')  # shape=(HIDDEN_DIMS,)
        latent_for_decoder3 = layers.Input(batch_shape=(BATCH_SIZE, HIDDEN_DIMS), name='dc_input_latent_3')  # shape=(HIDDEN_DIMS,)
        condition_for_decoder = layers.Input(batch_shape=(BATCH_SIZE, NUM_INFO), name='dc_input_cond')  # shape=(NUM_INFO,)

        dec_merged0 = layers.concatenate([latent_for_decoder0, condition_for_decoder])
        dec_d0 = self.decoder_dense0(dec_merged0)
        dec_d1 = self.decoder_dense1(dec_d0)
        dec_reshaped = layers.Reshape((INPUT_IMG_HEIGHT // 8, INPUT_IMG_WIDTH // 8, 256))(dec_d1)

        dec_merged1 = layers.concatenate([latent_for_decoder1, condition_for_decoder])
        dec_merged2 = layers.concatenate([latent_for_decoder2, condition_for_decoder])
        dec_merged3 = layers.concatenate([latent_for_decoder3, condition_for_decoder])

        dec_add_c0 = layers.Dense((INPUT_IMG_HEIGHT // 8) * (INPUT_IMG_WIDTH // 8) * 256, name='dec_c0_add', activation=silu_mul)(dec_merged1)
        dec_add_c0_ = layers.Reshape((INPUT_IMG_HEIGHT // 8, INPUT_IMG_WIDTH // 8, 256))(dec_add_c0)

        dec_add_c1 = layers.Dense((INPUT_IMG_HEIGHT // 4) * (INPUT_IMG_WIDTH // 4) * 128, name='dec_c1_add', activation=silu_mul)(dec_merged2)
        dec_add_c1_ = layers.Reshape((INPUT_IMG_HEIGHT // 4, INPUT_IMG_WIDTH // 4, 128))(dec_add_c1)

        dec_add_c2 = layers.Dense((INPUT_IMG_HEIGHT // 2) * (INPUT_IMG_WIDTH // 2) * 64, name='dec_c2_add', activation=silu_mul)(dec_merged3)
        dec_add_c2_ = layers.Reshape((INPUT_IMG_HEIGHT // 2, INPUT_IMG_WIDTH // 2, 64))(dec_add_c2)

        dec_add_c3 = layers.Dense(INPUT_IMG_HEIGHT * INPUT_IMG_WIDTH * 32, name='dec_c3_add', activation=silu_mul)(condition_for_decoder)
        dec_add_c3_ = layers.Reshape((INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 32))(dec_add_c3)

        # 6. decoder deconv CNN layers (decoder main stream)
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

        dec_final = layers.Reshape((INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, NUM_CHANNELS))(dec_c3)

        # 7. define encoder, decoder and cvae model
        self.encoder0 = tf.keras.Model([input_image, input_condition], self.latent0_space, name='encoder0')
        self.encoder1 = tf.keras.Model([input_image, input_condition], self.latent1_space, name='encoder1')
        self.encoder2 = tf.keras.Model([input_image, input_condition], self.latent2_space, name='encoder2')
        self.encoder3 = tf.keras.Model([input_image, input_condition], self.latent3_space, name='encoder3')
        self.decoder = tf.keras.Model([latent_for_decoder0, latent_for_decoder1, latent_for_decoder2, latent_for_decoder3, condition_for_decoder],
                                      dec_final,
                                      name='decoder')

        self.cvae = tf.keras.Model(
            inputs=[input_image, input_condition, condition_for_decoder],
            outputs=self.decoder([self.encoder0([input_image, input_condition]),
                                  self.encoder1([input_image, input_condition]),
                                  self.encoder2([input_image, input_condition]),
                                  self.encoder3([input_image, input_condition]), condition_for_decoder]),
            name='final_cvae'
        )

    def call(self, inputs, training):
        return self.cvae(inputs)


# C-VAE 모델 정의 및 반환
def define_cvae_model():
    optimizer = optimizers.Adam(0.0006, decay=1e-6)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    cvae_module = CVAE_Model(dropout_rate=0.25) # 실제 모델은 model.cvae

    return cvae_module, optimizer, scheduler_callback


# C-VAE 모델 학습 실시 및 모델 저장
# train_info = train_condition (N, 5)
def train_cvae_model(train_input, train_info):

    # to solve "You must feed a value for placeholder tensor {tensor_name} with dtype float and shape {shape}."
    tf.keras.backend.set_learning_phase(False)
    cvae_module, optimizer, scheduler_callback = define_cvae_model()
    cvae_module.cvae.compile(loss=cvae_module.vae_loss, optimizer=optimizer)

    # 학습 실시
    train_history = cvae_module.cvae.fit(
        [train_input, train_info, train_info], train_input,
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[scheduler_callback],
        shuffle=True
    )
    save_cvae_loss_log(train_history)

    # 모델 구조 표시
    show_model_summary(cvae_module)

    cvae_module.encoder3.save('models/cvae/cvae_encoder')
    cvae_module.decoder.save('models/cvae/cvae_decoder')
    cvae_module.cvae.save('models/cvae/cvae_entire')

    return cvae_module.encoder3, cvae_module.decoder, cvae_module.cvae