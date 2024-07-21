# from GAI-P2

import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import cv2


# 모델 반환 (with early stopping + lr_reduced)
def define_model(model_class, early_stopping_patience, lr_reduce_patience):
    optimizer = optimizers.Adam(0.001, decay=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=lr_reduce_patience)

    model = model_class()
    return model, optimizer, early_stopping, lr_reduced


def train_cnn_model(train_input, train_output, model_class, epochs=15, validation_split=0.1, loss='binary_crossentropy',
                    early_stopping_patience=5, lr_reduce_patience=2):

    """Training CNN model

    Args:
        train_input      (np.array)               : training input images
        train_output     (np.array)               : training output images
        model_class      (TensorFlow Model Class) : python class for model
        epochs           (int)                    : epochs for model training
        validation_split (float)                  : validation data split ratio for model training

    Outputs:
        cnn_model (TensorFlow Model) : trained CNN model
    """

    cnn_model, optimizer, early_stopping, lr_reduced = define_model(model_class,
                                                                    early_stopping_patience=early_stopping_patience,
                                                                    lr_reduce_patience=lr_reduce_patience)
    cnn_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    print(f'train input shape : {np.shape(train_input)}')
    print(f'train output shape : {np.shape(train_output)}')

    cnn_model.fit(
        train_input, train_output,
        callbacks=[early_stopping, lr_reduced],
        epochs=epochs,
        validation_split=validation_split
    )

    cnn_model.summary()
    return cnn_model


def load_first_1k(dir_path):
    """
    Load first 1,000 images from directory
    """

    imgs = sorted(os.listdir(dir_path))
    imgs_1k = imgs[:1000]
    first_1k_images = []

    for img in imgs_1k:
        img_path = dir_path + '/' + img
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        first_1k_images.append(img_cv2)

    first_1k_images = np.array(first_1k_images)
    first_1k_images = first_1k_images / 255.0
    return first_1k_images


def load_training_data_for_IDM(input_type, cropped_img_x_start, cropped_img_width, cropped_img_y_start, cropped_img_height):

    """
    Load training data for "Input Decision Models"

    Input:
        input_type          (str) : CVAE input type for Input Decision Models
                                    one of 'background', 'eyes', 'hair_color', 'head' or 'mouth'
        cropped_img_x_start (int) : X-axis value of left boundary of, cropped part of image to use for train
        cropped_img_width   (int) : width of, cropped part of image to use for train
        cropped_img_y_start (int) : Y-axis value of top boundary of, cropped part of image to use for train
        cropped_img_height  (int) : height of, cropped part of image to use for train

    Return:
        train_input_img (np.array) : input image for training
        train_output    (np.array) : output values (continuous value for regression)
    """

    # 각 성별의 처음 1,000장의 이미지에 대한 output values 와 해당 성별의 처음 1,000장의 이미지에 대해,
    # "이미지 조합 및 그 순서가 반드시 일치" 해야 함!!

    female_images = load_first_1k('dataset/resized/female')
    male_images = load_first_1k('dataset/resized/male')
    all_images = np.concatenate((female_images, male_images), axis=0)

    train_output_df = pd.read_csv(f'models/data/train_output_{input_type}.csv')
    train_output_df = train_output_df[['value']]

    train_input_img = np.array(all_images)
    train_input_img_cropped = train_input_img[:,
                                              cropped_img_y_start:cropped_img_y_start+cropped_img_height,
                                              cropped_img_x_start:cropped_img_x_start+cropped_img_width,
                                              :]
    train_output = np.array(train_output_df).reshape((len(train_output_df), 1))

    print(f'input type                    : {input_type}')
    print(f'shape of train input (images) : {np.shape(train_input_img_cropped)}')
    print(f'shape of train output         : {np.shape(train_output)}')

    return train_input_img_cropped, train_output