from model_decide_train_data import Classify_Male_Or_Female_CNN_Model, load_training_data, predict_male_or_female_for_all_images
from model_utils import train_cnn_model
import tensorflow as tf
import os
import pandas as pd
import shutil


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


if __name__ == '__main__':
    model = train_male_or_female_model(model_dir='models/decide_train_data')
    predict_male_or_female(model)
    save_final_training_data()

