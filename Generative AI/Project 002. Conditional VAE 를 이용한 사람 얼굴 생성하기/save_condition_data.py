import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os
import cv2
import pandas as pd


HAIR_COLOR_YXHW = {'y': 0, 'x': 0, 'h': 120, 'w': 120} # hair color 모델용 이미지 영역 (전체 영역)
MOUTH_YXHW = {'y': 60, 'x': 30, 'h': 60, 'w': 60} # mouth 모델용 이미지 영역
EYES_YXHW = {'y': 30, 'x': 30, 'h': 32, 'w': 60} # eyes 모델용 이미지 영역

model_hair_color = tf.keras.models.load_model('regression_hair_color')
model_mouth = tf.keras.models.load_model('regression_mouth')
model_eyes = tf.keras.models.load_model('regression_eyes')
model_background_mean = tf.keras.models.load_model('regression_background_mean')
model_background_std = tf.keras.models.load_model('regression_background_std')

gender_predictions_csv = 'male_or_female_classify_result_for_all_images.csv'
gender_predictions = pd.read_csv(gender_predictions_csv, index_col=0)


# 이미지에 대한 hair_color, mouth, eyes 모델 출력 결과 추가
def add_model_outputs(image_name, image_dir, model_output_result):
    img_path = image_dir + '/' + image_name
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img.reshape((1, img.shape[0], img.shape[1], 3)) # (120, 120, 3) -> (1, 120, 120, 3)
    img = img / 255.0

    # use gender prediction model output
    male_prob = gender_predictions[gender_predictions['image_path'] == img_path]['prob_male'].iloc[0]
    female_prob = gender_predictions[gender_predictions['image_path'] == img_path]['prob_female'].iloc[0]
    gender_prob = np.array([male_prob, female_prob])

    # get hair color prediction model output
    hc_y = HAIR_COLOR_YXHW['y']
    hc_x = HAIR_COLOR_YXHW['x']
    hc_h = HAIR_COLOR_YXHW['h']
    hc_w = HAIR_COLOR_YXHW['w']

    img_for_hair_color_model = np.array(img[:, hc_y:hc_y+hc_h, hc_x:hc_x+hc_w])
    input_for_hair_color_model = np.concatenate((img_for_hair_color_model.flatten(), gender_prob.flatten()))
    input_for_hair_color_model = np.array(input_for_hair_color_model).reshape((1, len(input_for_hair_color_model)))
    
    hair_color_model_output = float(model_hair_color(input_for_hair_color_model)[0][0])

    # get mouth prediction model output
    mo_y = MOUTH_YXHW['y']
    mo_x = MOUTH_YXHW['x']
    mo_h = MOUTH_YXHW['h']
    mo_w = MOUTH_YXHW['w']

    img_for_mouth_model = np.array(img[:, mo_y:mo_y+mo_h, mo_x:mo_x+mo_w])
    input_for_mouth_model = np.concatenate((img_for_mouth_model.flatten(), gender_prob.flatten()))
    input_for_mouth_model = np.array(input_for_mouth_model).reshape((1, len(input_for_mouth_model)))

    mouth_model_output = float(model_mouth(input_for_mouth_model)[0][0])

    # get eyes prediction model output
    ey_y = EYES_YXHW['y']
    ey_x = EYES_YXHW['x']
    ey_h = EYES_YXHW['h']
    ey_w = EYES_YXHW['w']

    img_for_eyes_model = np.array(img[:, ey_y:ey_y+ey_h, ey_x:ey_x+ey_w])
    input_for_eyes_model = np.concatenate((img_for_eyes_model.flatten(), gender_prob.flatten()))
    input_for_eyes_model = np.array(input_for_eyes_model).reshape((1, len(input_for_eyes_model)))

    eyes_model_output = float(model_eyes(input_for_eyes_model)[0][0])

    # get background mean/std prediction model output
    input_original = np.concatenate((img.flatten(), gender_prob.flatten()))
    input_original = np.array(input_original).reshape((1, len(input_original)))

    background_mean_model_output = float(model_background_mean(input_original)[0][0])
    background_std_model_output = float(model_background_std(input_original)[0][0])

    # add output result
    output_result = {
        'image_path': [img_path],
        'male_prob': [male_prob],
        'female_prob': [female_prob],
        'hair_color': [hair_color_model_output],
        'mouth': [mouth_model_output],
        'eyes': [eyes_model_output],
        'background_mean': [background_mean_model_output],
        'background_std': [background_std_model_output]
    }
    
    output_result = pd.DataFrame(output_result)
    model_output_result = pd.concat([model_output_result, output_result])

    if len(model_output_result) % 250 == 0:
        print(len(model_output_result))

    return model_output_result


# 모든 이미지에 대한 hair color, mouth, eyes 모델 출력값 저장
def save_model_output_for_all_images():
    print('prediction start ...')
    
    model_output_result = pd.DataFrame()

    image_dirs = ['resized_images/first_dataset', 'resized_images/second_dataset_male', 'resized_images/second_dataset_female']

    for image_dir in image_dirs:
        for image_name in os.listdir(image_dir):
            model_output_result = add_model_outputs(image_name, image_dir, model_output_result)

    model_output_result.to_csv('condition_data.csv')
    print('finished')


if __name__ == '__main__':
    save_model_output_for_all_images()
