from model_decide_train_data import Classify_Male_Or_Female_CNN_Model, load_training_data, predict_male_or_female_for_all_images
from model_utils import train_cnn_model, load_training_data_for_IDM
import tensorflow as tf
import os
import pandas as pd
import shutil
import cv2
import numpy as np

from model_input_background import Regression_Background_Model
from model_input_eyes import Regression_Eyes_Model
from model_input_hair_color import Regression_Hair_Color_Model
from model_input_head import Regression_Head_Model
from model_input_mouth import Regression_Mouth_Model


# Input Decision Model name to TensorFlow Model class mapping
IDL_CLASS_MAPPING = {
    'background': Regression_Background_Model,
    'eyes': Regression_Eyes_Model,
    'hair_color': Regression_Hair_Color_Model,
    'head': Regression_Head_Model,
    'mouth': Regression_Mouth_Model
}

# Input Decision Model name, to the resolution info of cropped part of images to train IDM
IMG_INFO_MAPPING = {
    'background': {'x_start': 0,  'width': 104, 'y_start': 0,  'height': 128},
    'eyes':       {'x_start': 24, 'width': 56,  'y_start': 48, 'height': 24},
    'hair_color': {'x_start': 0,  'width': 104, 'y_start': 0,  'height': 128},
    'head':       {'x_start': 0,  'width': 104, 'y_start': 0,  'height': 128},
    'mouth':      {'x_start': 36, 'width': 32,  'y_start': 88, 'height': 16},
}


def train_male_or_female_model(model_dir):

    """
    Train "Training Data Decision Model (학습 데이터 결정 모델)"

    Args:
        model_dir (str) : directory to save model
    """

    if os.path.exists(model_dir):
        print('male/female model already exists')
        model = tf.keras.models.load_model(model_dir)
        return model

    print('loading training data ...')
    train_input, train_output = load_training_data('resized')

    print('loading cnn model ...')
    model = train_cnn_model(train_input, train_output, Classify_Male_Or_Female_CNN_Model)
    model.save(model_dir)

    return model


def predict_male_or_female(model):
    """
    Predict Male/Female using "Training Data Decision Model", for entire dataset

    Args:
        model (TensorFlow model) : saved "Training Data Decision Model"

    File Outputs:
        models/all_output_decide_train_data.csv : Male/Female prediction result for entire dataset,
                                                  including model's training data
    """

    if os.path.exists('models/all_output_decide_train_data.csv'):
        print('prediction result already exists')
        return

    print('prediction start ...')
    all_data_dir_list = ['augmented/10k-images', 'augmented/female', 'augmented/male',
                         'resized/10k-images', 'resized/female', 'resized/male']

    predict_male_or_female_for_all_images(model, all_data_dir_list)


def save_final_training_data():
    """
    Save final training data for NVAE-idea-based CVAE (main model)
    """

    if os.path.exists('final'):
        print('final training data for CVAE already exists')
        return

    prediction_result = pd.read_csv('models/all_output_decide_train_data.csv', index_col=0)
    prediction_result = prediction_result[prediction_result['prob_female'] >= 0.9999]
    prediction_result = prediction_result[~prediction_result['image_path'].str.contains('/male/')]
    print('final training data for CVAE :\n', prediction_result)

    os.makedirs('final', exist_ok=True)

    for _, row in prediction_result.iterrows():
        img_path = row['image_path']
        aug_or_res, dir_name, file_name = img_path.split('/')[0], img_path.split('/')[1], img_path.split('/')[2]
        new_img_path = f'final/{aug_or_res}_{dir_name}_{file_name}'
        shutil.copyfile(img_path, new_img_path)


def train_input_decision_models_for_input_type(input_type):
    """
    Generate and then train "Input Decision Model" for specific input type

    Args:
        input_type (str) : type indicating each 'CVAE input value' for training Input Decision Models
                           one of 'background', 'eyes', 'hair_color', 'head' or 'mouth'

    Return:
        input_decision_model (TensorFlow Model) : trained "Input Decision Model"
    """

    if os.path.exists(f'models/input_{input_type}'):
        print(f'Input Decision Model for {input_type} already exists')
        return

    train_input_image, train_output = load_training_data_for_IDM(input_type=input_type,
                                                                 cropped_img_x_start=IMG_INFO_MAPPING[input_type]['x_start'],
                                                                 cropped_img_width=IMG_INFO_MAPPING[input_type]['width'],
                                                                 cropped_img_y_start=IMG_INFO_MAPPING[input_type]['y_start'],
                                                                 cropped_img_height=IMG_INFO_MAPPING[input_type]['height'])

    # 기본 옵션 (early_stopping_patience=5, lr_reduce_patience=2) 로 학습이 잘 안되는 eyes 의 경우, patience 관련 특별 처리
    # eyes 의 경우 학습 종료시점 기준 valid loss 가 0.035 정도 수준이면 안되고, 0.020 정도여야 학습이 잘 된 것임
    # (약 50% 확률로 학습이 잘되는 것으로 추정)
    input_decision_model = train_cnn_model(train_input=train_input_image,
                                           train_output=train_output,
                                           model_class=IDL_CLASS_MAPPING[input_type],
                                           loss='mse',
                                           epochs=30,
                                           early_stopping_patience=(12 if input_type == 'eyes' else 5),
                                           lr_reduce_patience=(30 if input_type == 'eyes' else 2))

    return input_decision_model


def generate_input_decision_models():
    """
    Generate and then train Input Decision Models

    File Outputs:
        input_background (TensorFlow Model directory) : Input Decision Models for 'background'
        input_eyes       (TensorFlow Model directory) : Input Decision Models for 'eyes'
        input_hair_color (TensorFlow Model directory) : Input Decision Models for 'hair_color'
        input_head       (TensorFlow Model directory) : Input Decision Models for 'head'
        input_mouth      (TensorFlow Model directory) : Input Decision Models for 'mouth'
    """

    input_types = ['background', 'eyes', 'hair_color', 'head', 'mouth']
    for input_type in input_types:
        input_decision_model = train_input_decision_models_for_input_type(input_type)

        if input_decision_model is not None:
            input_decision_model.save(f'models/input_{input_type}')


def predict_with_input_decision_models_for_input_type(input_type):
    """
    Predict continuous values using "Input Decision Model" for specific input type, for entire final training dataset

    Args:
        input_type (str) : type indicating each 'CVAE input value' for prediction
                           one of 'background', 'eyes', 'hair_color', 'head' or 'mouth'

    File Output:
        all_output_{input_type}.csv : prediction result for entire final training dataset
    """

    if os.path.exists(f'models/all_output_{input_type}.csv'):
        print(f'prediction result (by Input Decision Model) for {input_type} already exists')
        return

    input_decision_model = tf.keras.models.load_model(f'models/input_{input_type}')

    final_training_img_names = os.listdir('final')
    img_paths = []
    predicted_values = []

    for idx, img_name in enumerate(final_training_img_names):
        if idx % 250 == 0:
            print(f'progress (for {input_type}) : {idx}')

        img_path = 'final/' + img_name
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img.reshape((1, img.shape[0], img.shape[1], 3))  # (128, 104, 3) -> (1, 128, 104, 3)

        cropped_img_x_start = IMG_INFO_MAPPING[input_type]['x_start']
        cropped_img_width = IMG_INFO_MAPPING[input_type]['width']
        cropped_img_y_start = IMG_INFO_MAPPING[input_type]['y_start']
        cropped_img_height = IMG_INFO_MAPPING[input_type]['height']

        img_cropped = img[:,
                          cropped_img_y_start:cropped_img_y_start + cropped_img_height,
                          cropped_img_x_start:cropped_img_x_start + cropped_img_width,
                          :]
        img_cropped = img_cropped / 255.0

        prediction = np.array(input_decision_model(img_cropped))

        img_paths.append(img_path)
        predicted_values.append(prediction[0][0])

    prediction_result = {'image_path': img_paths, input_type: predicted_values}
    prediction_result_df = pd.DataFrame(prediction_result)
    prediction_result_df.to_csv(f'models/all_output_{input_type}.csv', index=False)


def predict_with_input_decision_models():
    """
    Predict continuous values using "Input Decision Model", for entire final training dataset
    """

    input_types = ['background', 'eyes', 'hair_color', 'head', 'mouth']
    for input_type in input_types:
        predict_with_input_decision_models_for_input_type(input_type)



if __name__ == '__main__':
    model = train_male_or_female_model(model_dir='models/decide_train_data')
    predict_male_or_female(model)
    save_final_training_data()

    # generate and predict all images with "Input Decision Models" (입력값 결정 모델)
    generate_input_decision_models()
    predict_with_input_decision_models()
