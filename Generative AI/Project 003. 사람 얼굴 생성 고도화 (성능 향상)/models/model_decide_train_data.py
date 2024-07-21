# from GAI-P2

import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd


class Classify_Male_Or_Female_CNN_Model(tf.keras.Model):

    def __init__(self, dropout_rate=0.25):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.pooling = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')
        L2 = tf.keras.regularizers.l2(0.001)

        # conv + pooling part
        # height : 128 -> 126 -> 124 -> 62 -> 60 -> 30 -> 28 -> 14 -> 12
        # width  : 104 -> 102 -> 100 -> 50 -> 48 -> 24 -> 22 -> 11 ->  9
        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[120, 120, 3])
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
        inputs = tf.keras.layers.Reshape((128, 104, 3))(inputs)

        # conv + pooling part
        # height : 128 -> 126 -> 124 -> 62 -> 60 -> 30 -> 28 -> 14 -> 12
        # width  : 104 -> 102 -> 100 -> 50 -> 48 -> 24 -> 22 -> 11 ->  9
        outputs_0 = self.conv_0(inputs)
        outputs_1 = self.conv_1(outputs_0)
        outputs_2 = self.pooling(outputs_1)

        outputs_3 = self.conv_2(outputs_2)
        outputs_4 = self.pooling(outputs_3)

        outputs_5 = self.conv_3(outputs_4)
        outputs_6 = self.pooling(outputs_5)

        outputs_7 = self.conv_4(outputs_6)
        outputs_flatten = self.flatten(outputs_7)

        # fully connected part
        dense = self.dense_0(outputs_flatten)
        dense = self.dropout(dense)
        dense = self.dense_1(dense)
        dense = self.dropout(dense)
        final_output = self.final_dense(dense)

        return final_output


def load_training_data(data_dir):

    """Load Training Data

    Args:
        data_dir (str) : directory including Male and Female images from https://thispersondoesnotexist.com/

    Outputs:
        train_input  (np.array) : training input images
        train_output (np.array) : training output images
    """

    male_images_name = os.listdir(f'{data_dir}/male')
    female_images_name = os.listdir(f'{data_dir}/female')

    train_input = []
    train_output = []

    # for male images
    for idx, name in enumerate(male_images_name):
        if idx % 500 == 0:
            print(f'progress : male image {idx}')

        img = cv2.imread(f'{data_dir}/male/' + name, cv2.IMREAD_UNCHANGED)
        train_input.append(np.array(img) / 255.0)
        train_output.append([1, 0])

    # for female images
    for idx, name in enumerate(female_images_name):
        if idx % 500 == 0:
            print(f'progress : female image {idx}')

        img = cv2.imread(f'{data_dir}/female/' + name, cv2.IMREAD_UNCHANGED)
        train_input.append(np.array(img) / 255.0)
        train_output.append([0, 1])

    train_input = np.array(train_input)
    train_output = np.array(train_output)

    print(f'shape of train input : {np.shape(train_input)}')
    print(f'shape of train output : {np.shape(train_output)}')

    return train_input, train_output


def add_predict_results(image_name, image_dir, predict_result_df, cnn_model):

    """Predict gender for each image

    Args:
        image_name        (str)              : name of image (excluding path)
        image_dir         (str)              : directory including the image (with path)
        predict_result_df (Pandas DataFrame) : Pandas DataFrame with already existing gender prediction results
        cnn_model         (TensorFlow Model) : CNN model

    Outputs:
        predict_result_df (Pandas DataFrame) : Pandas DataFrame with already existing + new gender prediction results
    """

    img_path = image_dir + '/' + image_name

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img.reshape((1, img.shape[0], img.shape[1], 3))  # (128, 104, 3) -> (1, 128, 104, 3)
    img = img / 255.0

    prediction = np.array(cnn_model(img))

    prediction_result = {'image_path': [img_path], 'prob_male': [prediction[0][0]], 'prob_female': [prediction[0][1]]}
    prediction_result = pd.DataFrame(prediction_result)
    predict_result_df = pd.concat([predict_result_df, prediction_result])

    if len(predict_result_df) % 250 == 0:
        print(f'prediction progress : {len(predict_result_df)}')

    return predict_result_df


def predict_male_or_female_for_all_images(cnn_model, image_dirs):

    """Predict gender all the images in directories

    Args:
        image_dirs (list)             : list of directory names including the image (including path)
        cnn_model  (TensorFlow Model) : CNN model

    Outputs:
        train_input (np.array)  : training input images
        train_output (np.array) : training output images
    """

    predict_result = pd.DataFrame()

    for image_dir in image_dirs:
        print(f'\nnumber of images in {image_dir} : {len(os.listdir(image_dir))}')
        for image_name in os.listdir(image_dir):
            predict_result = add_predict_results(image_name, image_dir, predict_result, cnn_model)

    predict_result.to_csv('models/all_output_decide_train_data.csv')
