import cv2
import os
import numpy as np
import pandas as pd


# 특정 디렉토리의 처음 1,000장 읽어오기
def load_first_1k(dir_path):
    imgs = sorted(os.listdir(dir_path))
    imgs_1k = imgs[:1000]
    first_1k_images = []

    for img in imgs_1k:
        img_path = dir_path + '/' + img
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        first_1k_images.append(img_cv2)

    first_1k_images = np.array(first_1k_images)
    first_1k_images = first_1k_images / 255.0
    return first_1k_images, imgs_1k


# read output values
def read_output(csv_name):
    df = pd.read_csv(csv_name, index_col=0)
    return np.array(df).reshape(len(df), 1)


# read gender probability info
def read_gender_prob():
    df = pd.read_csv('male_or_female_classify_result_for_all_images.csv', index_col=0)
    df_dict = {}

    for idx, row in df.iterrows():
        img_path = row['image_path']
        prob_male = row['prob_male']
        prob_female = row['prob_female']

        df_dict[img_path] = {'prob_male': prob_male, 'prob_female': prob_female}

    return df_dict


# 각 성별의 처음 1,000장의 이미지에 대한 output values 와 해당 성별의 처음 1,000장의 이미지에 대해,
# "이미지 조합 및 그 순서가 반드시 일치" 해야 함!!
if __name__ == '__main__':
    first_1k_images_male, male_img_names = load_first_1k('resized_images/second_dataset_male')
    first_1k_images_female, female_img_names = load_first_1k('resized_images/second_dataset_female')

    print(np.shape(first_1k_images_male))
    print(np.shape(first_1k_images_female))
    
    print('\nmale image names :')
    print(male_img_names[:20])

    print('\nfemale image names :')
    print(female_img_names[:20])

    gender_prob = read_gender_prob()

    print('\ngender probabilities :')
    print(str(gender_prob)[:1000])
