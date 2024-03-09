import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class Main_CNN_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # common convolution part
        self.conv_common_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_common_0', input_shape=[14, 14, 1])
        self.conv_common_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_common_1')

        # splitted convolution part
        self.conv_split0_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_split0_0')
        self.conv_split0_1 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', name='conv_split0_1')
        self.conv_split0_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_split0_2')

        self.conv_split1_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_split1_0')
        self.conv_split1_1 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', name='conv_split1_1')
        self.conv_split1_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_split1_2')

        self.conv_split2_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_split2_0')
        self.conv_split2_1 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', name='conv_split2_1')
        self.conv_split2_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_split2_2')

        self.conv_split3_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_split3_0')
        self.conv_split3_1 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', name='conv_split3_1')
        self.conv_split3_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_split3_2')

        self.conv_split4_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv_split4_0')
        self.conv_split4_1 = tf.keras.layers.Conv2D(48, (3, 3), activation='relu', name='conv_split4_1')
        self.conv_split4_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_split4_2')

        # fully connected part (for output A)
        self.dense0_0 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense0_0')

        self.dense0_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense0_1')

        self.dense0_2 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                              kernel_regularizer=L2, name='dense0_2')

        self.dense0_final = tf.keras.layers.Dense(1, activation='sigmoid',
                                                  kernel_regularizer=L2, name='dense0_final')

        # fully connected part (for output B)
        self.dense1_0 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense1_0')

        self.dense1_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense1_1')

        self.dense1_2 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                              kernel_regularizer=L2, name='dense1_2')

        self.dense1_final = tf.keras.layers.Dense(1, activation='sigmoid',
                                                  kernel_regularizer=L2, name='dense1_final')

        # fully connected part (for output C)
        self.dense2_0 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense2_0')

        self.dense2_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense2_1')

        self.dense2_2 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                              kernel_regularizer=L2, name='dense2_2')

        self.dense2_final = tf.keras.layers.Dense(1, activation='sigmoid',
                                                  kernel_regularizer=L2, name='dense2_final')

        # fully connected part (for output D)
        self.dense3_0 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense3_0')

        self.dense3_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense3_1')

        self.dense3_2 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                              kernel_regularizer=L2, name='dense3_2')

        self.dense3_final = tf.keras.layers.Dense(1, activation='sigmoid',
                                                  kernel_regularizer=L2, name='dense3_final')

        # fully connected part (for output E)
        self.dense4_0 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense4_0')

        self.dense4_1 = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.05),
                                              kernel_regularizer=L2, name='dense4_1')

        self.dense4_2 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.025),
                                              kernel_regularizer=L2, name='dense4_2')

        self.dense4_final = tf.keras.layers.Dense(1, activation='sigmoid',
                                                  kernel_regularizer=L2, name='dense4_final')


    def call(self, inputs, training):
        inputs_cnn, inputs_center_4 = tf.split(inputs, [14 * 14, 4], axis=1)
        inputs_cnn = tf.keras.layers.Reshape((14, 14, 1))(inputs_cnn)

        # common convolution part : 14 -> 12 -> 10
        outputs_0 = self.conv_common_0(inputs_cnn)
        outputs_1 = self.conv_common_1(outputs_0)

        # split covolution part : 10 -> 8 -> 6 -> 4
        outputs_00 = self.conv_split0_0(outputs_1)
        outputs_01 = self.conv_split0_1(outputs_00)
        outputs_02 = self.conv_split0_2(outputs_01)

        outputs_10 = self.conv_split1_0(outputs_1)
        outputs_11 = self.conv_split1_1(outputs_10)
        outputs_12 = self.conv_split1_2(outputs_11)

        outputs_20 = self.conv_split2_0(outputs_1)
        outputs_21 = self.conv_split2_1(outputs_20)
        outputs_22 = self.conv_split2_2(outputs_21)

        outputs_30 = self.conv_split3_0(outputs_1)
        outputs_31 = self.conv_split3_1(outputs_30)
        outputs_32 = self.conv_split3_2(outputs_31)

        outputs_40 = self.conv_split4_0(outputs_1)
        outputs_41 = self.conv_split4_1(outputs_40)
        outputs_42 = self.conv_split4_2(outputs_41)

        # fully connected part for output A
        flatten_0 = self.flatten(outputs_02)
        
        dense_00 = self.dense0_0(flatten_0)
        dense_00 = self.dropout(dense_00)
        dense_00 = tf.keras.layers.concatenate([dense_00, inputs_center_4])
        
        dense_01 = self.dense0_1(dense_00)
        dense_01 = self.dropout(dense_01)
        dense_01 = tf.keras.layers.concatenate([dense_01, inputs_center_4])

        dense_02 = self.dense0_2(dense_01)
        dense_02 = self.dropout(dense_02)
        dense_02 = tf.keras.layers.concatenate([dense_02, inputs_center_4])
        
        dense_0_final = self.dense0_final(dense_02)

        # fully connected part for output B
        flatten_1 = self.flatten(outputs_12)
        
        dense_10 = self.dense1_0(flatten_1)
        dense_10 = self.dropout(dense_10)
        dense_10 = tf.keras.layers.concatenate([dense_10, inputs_center_4])
        
        dense_11 = self.dense1_1(dense_10)
        dense_11 = self.dropout(dense_11)
        dense_11 = tf.keras.layers.concatenate([dense_11, inputs_center_4])
        
        dense_12 = self.dense1_2(dense_11)
        dense_12 = self.dropout(dense_12)
        dense_12 = tf.keras.layers.concatenate([dense_12, inputs_center_4])
        
        dense_1_final = self.dense1_final(dense_12)

        # fully connected part for output C
        flatten_2 = self.flatten(outputs_22)
        
        dense_20 = self.dense2_0(flatten_2)
        dense_20 = self.dropout(dense_20)
        dense_20 = tf.keras.layers.concatenate([dense_20, inputs_center_4])
        
        dense_21 = self.dense2_1(dense_20)
        dense_21 = self.dropout(dense_21)
        dense_21 = tf.keras.layers.concatenate([dense_21, inputs_center_4])
        
        dense_22 = self.dense2_2(dense_21)
        dense_22 = self.dropout(dense_22)
        dense_22 = tf.keras.layers.concatenate([dense_22, inputs_center_4])
        
        dense_2_final = self.dense2_final(dense_22)

        # fully connected part for output D
        flatten_3 = self.flatten(outputs_32)
        
        dense_30 = self.dense3_0(flatten_3)
        dense_30 = self.dropout(dense_30)
        dense_30 = tf.keras.layers.concatenate([dense_30, inputs_center_4])
        
        dense_31 = self.dense3_1(dense_30)
        dense_31 = self.dropout(dense_31)
        dense_31 = tf.keras.layers.concatenate([dense_31, inputs_center_4])
        
        dense_32 = self.dense3_2(dense_31)
        dense_32 = self.dropout(dense_32)
        dense_32 = tf.keras.layers.concatenate([dense_32, inputs_center_4])
        
        dense_3_final = self.dense3_final(dense_32)

        # fully connected part for output E
        flatten_4 = self.flatten(outputs_42)
        
        dense_40 = self.dense4_0(flatten_4)
        dense_40 = self.dropout(dense_40)
        dense_40 = tf.keras.layers.concatenate([dense_40, inputs_center_4])
        
        dense_41 = self.dense4_1(dense_40)
        dense_41 = self.dropout(dense_41)
        dense_41 = tf.keras.layers.concatenate([dense_41, inputs_center_4])
        
        dense_42 = self.dense4_2(dense_41)
        dense_42 = self.dropout(dense_42)
        dense_42 = tf.keras.layers.concatenate([dense_42, inputs_center_4])
        
        dense_4_final = self.dense4_final(dense_42)
        
        # final concatenation layer
        final_output = tf.keras.layers.concatenate([dense_0_final, dense_1_final, dense_2_final, dense_3_final, dense_4_final])
        return final_output
    

# 모델 반환
def define_model():
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2)
        
    model = Main_CNN_Model()
    return model, optimizer, early_stopping, lr_reduced


# 모델 학습
def train_main_model(train_input, train_output):
    cnn_model, optimizer, early_stopping, lr_reduced = define_model()
    cnn_model.compile(loss='mse', optimizer=optimizer)

    cnn_model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=5,
        validation_split=0.1
    )

    cnn_model.summary()
    return cnn_model


# 학습 데이터 불러오기
def create_train_data():
    train_data = np.array(pd.read_csv('train_data.csv', index_col=0)) / 255.0

    train_input_cnn = train_data[:, :14 * 14]
    train_input_center = train_data[:, [90, 91, 104, 105]]
    train_input = np.concatenate([train_input_cnn, train_input_center], axis=1)
    
    train_output = train_data[:, 14 * 14:]

    print(f'train input  (CNN)    : {np.shape(train_input_cnn)}\n{train_input_cnn}\n')
    print(f'train input  (center) : {np.shape(train_input_center)}\n{train_input_center}\n')
    print(f'train input  (all)    : {np.shape(train_input)}\n{train_input}\n')
    
    print(f'train output          : {np.shape(train_output)}\n{train_output}\n')

    return train_input, train_output


if __name__ == '__main__':

    # 학습 데이터 받아오기
    train_input, train_output = create_train_data()    

    # 메인 CNN 모델 학습
    model = train_main_model(train_input, train_output)

    # 메인 CNN 모델 저장
    model.save('main_model')

