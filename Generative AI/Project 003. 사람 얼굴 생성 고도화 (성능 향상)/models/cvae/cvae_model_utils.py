# from GAI-P2

import pandas as pd
import numpy as np
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

BATCH_SIZE = 32  # same as 'cvae_model_architecture.py'


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


# model architecture image
def plot_model_architecture(model, img_file_path):
    try:
        tf.keras.utils.plot_model(model, to_file=img_file_path, show_shapes=True)
    except Exception as e:
        print(f'model architecture image file generation error : {e}')


# 모델 구조 표시
def show_model_summary(cvae_module):
    architecture_img_dir = 'models/model_architecture'
    os.makedirs(architecture_img_dir, exist_ok=True)

    print('\n === ENCODER ===')
    cvae_module.encoder3.summary()
    plot_model_architecture(cvae_module.encoder0, os.path.join(architecture_img_dir, 'encoder0.png'))
    plot_model_architecture(cvae_module.encoder1, os.path.join(architecture_img_dir, 'encoder1.png'))
    plot_model_architecture(cvae_module.encoder2, os.path.join(architecture_img_dir, 'encoder2.png'))
    plot_model_architecture(cvae_module.encoder3, os.path.join(architecture_img_dir, 'encoder3.png'))

    print('\n === DECODER ===')
    cvae_module.decoder.summary()
    plot_model_architecture(cvae_module.decoder, os.path.join(architecture_img_dir, 'decoder.png'))

    print('\n === VAE ===')
    cvae_module.cvae.summary()
    plot_model_architecture(cvae_module.cvae, os.path.join(architecture_img_dir, 'cvae.png'))


# C-VAE 모델 학습 loss 기록 저장
def save_cvae_loss_log(train_history):
    plt.plot(train_history.history['loss'])
    plt.title('CVAE loss history')
    plt.xlabel('epoch')
    plt.ylabel('total loss')
    plt.savefig('cvae_train_result.png')


def create_train_and_valid_data(limit=None):

    """
    Get train dataset and additional info (background, eyes, hair color, head, mouth) for CVAE

    Inputs:
        limit (int) : number of train data rows limit, for test

    Return:
        train_input (np.array) : input image
        train_info  (np.array) : additional info (background, eyes, hair color, head, mouth) for CVAE
    """

    condition_data = pd.read_csv('models/data/all_output.csv')
    print(condition_data)

    train_input = []
    train_info = []
    current_idx = 0

    for _, row in condition_data.iterrows():
        if current_idx % 250 == 0:
            print(current_idx)

        img_path = row['image_paths']

        # condition info (total 5)
        background = row['background']
        eyes = row['eyes']
        hair_color = row['hair_color']
        head = row['head']
        mouth = row['mouth']

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        train_input.append(np.array(img) / 255.0)
        train_info.append([background, eyes, hair_color, head, mouth])

        current_idx += 1
        if limit is not None and current_idx >= limit:
            break

    # batch size is 32 -> only "multiple of 32" is available total dataset size
    available_length = len(train_input) // BATCH_SIZE * BATCH_SIZE
    print(f'total dataset size : {available_length}')

    train_input = np.array(train_input)[:available_length]
    train_info = np.array(train_info)[:available_length]

    return train_input, train_info