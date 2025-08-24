
import os
import shutil
import random

import pandas as pd
from PIL import Image


TEST_SPLIT_RATIO = 0.2


# Scanned Images Dataset for OCR and VLM finetuning 데이터셋을 train 80% : test 20% 로 분리
# Create Date : 2025.08.24
# Last Update Date : -

# Arguments:
# - 없음

def split_train_and_test_data():
    img_paths = {
        'letter': 'scanned_images_dataset/dataset/Letter',
        'memo': 'scanned_images_dataset/dataset/Memo',
        'report': 'scanned_images_dataset/dataset/Report'
    }

    img_names, test_idxs, train_test_split_info = {}, {}, {}

    # split train & test data
    for k, v in img_paths.items():
        img_names[k] = os.listdir(v)
        img_count = len(img_names[k])
        test_idxs[k] = random.sample(list(range(img_count)), int(TEST_SPLIT_RATIO * img_count))

        train_test_split_info[k] = ['train' for _ in range(img_count)]
        for test_idx in test_idxs[k]:
            train_test_split_info[k][test_idx] = 'test'

    # copy images
    for img_type in img_paths.keys():
        train_data_dir_name = img_paths[img_type].replace('/dataset/', '/train/')
        test_data_dir_name = img_paths[img_type].replace('/dataset/', '/test/')

        os.makedirs(train_data_dir_name, exist_ok=True)
        os.makedirs(test_data_dir_name, exist_ok=True)

        for idx, (img_name, split_info) in enumerate(zip(img_names[img_type], train_test_split_info[img_type])):
            src = str(os.path.join(img_paths[img_type], img_name))

            if train_test_split_info[img_type][idx] == 'train':
                dst = str(os.path.join(train_data_dir_name, img_name))
            else:  # test
                dst = str(os.path.join(test_data_dir_name, img_name))

            shutil.copy(src, dst)


# Scanned Images Dataset for OCR and VLM finetuning 데이터셋을 랜덤 각도로 회전
# Create Date : 2025.08.24
# Last Update Date : -

# Arguments:
# - 없음

# Output:
# - scanned_images_dataset/angle_label.csv : 회전 각도 label 파일

def apply_random_rotate():
    angle_info_dict = {'img_path': [], 'angle': []}
    angle_info_df_path = 'scanned_images_dataset/angle_label.csv'

    # image path dictionary
    img_paths = {
        'train': {
            'letter': 'scanned_images_dataset/train/Letter',
            'memo': 'scanned_images_dataset/train/Memo',
            'report': 'scanned_images_dataset/train/Report'
        },
        'test': {
            'letter': 'scanned_images_dataset/test/Letter',
            'memo': 'scanned_images_dataset/test/Memo',
            'report': 'scanned_images_dataset/test/Report'
        }
    }

    # image split result
    for img_split in img_paths.keys():
        print(f'rotate & copy images from {img_split} data ...')

        for path in img_paths[img_split].values():
            img_name_list = os.listdir(path)
            img_path_list = [os.path.join(path, img_name) for img_name in img_name_list]

            rotated_img_save_dir = path.replace(f'/{img_split}/', f'/{img_split}_rotated/')
            os.makedirs(rotated_img_save_dir, exist_ok=True)

            for img_name, img_path in zip(img_name_list, img_path_list):
                img = Image.open(img_path)
                angle = random.random() * 30.0 - 15.0  # -15 ~ +15 도 범위 회전
                rotated_img = img.rotate(angle, expand=True, fillcolor='white')

                save_path = os.path.join(rotated_img_save_dir, img_name)
                rotated_img.save(save_path)

                angle_info_dict['img_path'].append(save_path)
                angle_info_dict['angle'].append(round(angle, 2))

    # save angle info as Pandas DataFrame
    angle_info_df = pd.DataFrame(angle_info_dict)
    angle_info_df.to_csv(angle_info_df_path)


if __name__ == '__main__':
    print('data splitting ...')
    split_train_and_test_data()

    apply_random_rotate()

