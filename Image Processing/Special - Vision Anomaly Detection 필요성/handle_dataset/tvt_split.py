import os
import cv2
import numpy as np

import random
random.seed(2025)
from random import shuffle

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


# 전체 데이터셋의 이미지 resize 실시
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - img_dir   (str) : 이미지가 저장된 디렉토리 이름
# - dest_size (int) : resize 이후의 이미지 가로/세로 길이 (default: 512)

# Returns:
# - mvtec_dataset 디렉토리 안의 모든 이미지를 dest_size x dest_size 로 resize

def resize_all_images(img_dir, dest_size=512):
    img_files = []

    # 이미지 찾기
    for (path, dir, files) in os.walk(f'{PROJECT_DIR_PATH}/{img_dir}/'):
        for file_name in files:
            ext = os.path.splitext(file_name)[-1]
            if ext == '.png':
                img_files.append(f'{path}/{file_name}')

    # 이미지 resize
    for idx, img_file_path in enumerate(img_files):
        if idx % 100 == 0:
            print(f'resizing : {idx} / {len(img_files)}')

        is_mask = img_file_path.endswith('mask.png')

        # 이미지 읽기 시 한글 경로 처리
        if is_mask:
            img = cv2.imdecode(np.fromfile(img_file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(np.fromfile(img_file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # already resized -> pass
        if np.shape(img)[:2] == (dest_size, dest_size):
            continue

        if is_mask:
            resized_img = cv2.resize(img, (dest_size, dest_size), interpolation=cv2.INTER_NEAREST)
        else:
            resized_img = cv2.resize(img, (dest_size, dest_size), interpolation=cv2.INTER_AREA)

        # 이미지 저장 시 한글 경로 처리
        result, img_arr = cv2.imencode(ext='.png', img=resized_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

        if result:
            with open(img_file_path, mode='w+b') as f:
                img_arr.tofile(f)

    print('resize finished')


# 원본 데이터셋의 데이터를 여러 그룹으로 분리하여 각 그룹에 해당하는 이미지 경로 반환
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - dataset_type  (str)         : 'train' or 'test'
# - dataset_class (str)         : 'normal' or 'abnormal'
# - category_list (list(str))   : 세부 카테고리 목록
# - group_names   (list(str))   : 그룹 이름 목록
# - split_ratio   (list(float)) : 각 그룹에 할당할 이미지의 비율

# Returns:
# - image_paths_per_group (dict(list)) : 각 그룹에 해당하는 이미지의 리스트를 저장한 dict

def split_dataset_into_groups(dataset_type, dataset_class, category_list, group_names, split_ratio):
    assert sum(split_ratio) == 1.0, "SUM OF SPLIT RATIO MUST BE 1.0"
    assert len(group_names) == len(split_ratio), "LENGTH OF group_names MUST BE SAME WITH LENGTH OF split_ratio"

    image_path_list = []

    for category in category_list:
        dataset_dir = f'{PROJECT_DIR_PATH}/mvtec_dataset_512/{category}/{dataset_type}'

        if dataset_class == 'normal':
            img_dir_path = f'{dataset_dir}/good'
            img_paths = os.listdir(img_dir_path)
            img_paths = [f'{img_dir_path}/{x}' for x in img_paths]
        else:
            img_dirs = list(filter(lambda x: x != 'good', os.listdir(dataset_dir)))
            img_paths = []

            for img_dir in img_dirs:
                img_dir_path = f'{dataset_dir}/{img_dir}'
                img_path_list = os.listdir(img_dir_path)
                img_path_list = [f'{img_dir_path}/{x}' for x in img_path_list]

                img_paths += img_path_list

        image_path_list += img_paths

    # apply random shuffle
    shuffle(image_path_list)

    image_cnt = len(image_path_list)
    current_cum_ratio = 0.0
    image_paths_per_group = {}

    for group_name, part_ratio in zip(group_names, split_ratio):

        part_start_idx = int(image_cnt * current_cum_ratio)
        part_end_idx = int(image_cnt * (current_cum_ratio + part_ratio))

        image_path_list_part = image_path_list[part_start_idx:part_end_idx]
        image_paths_per_group[group_name] = image_path_list_part

        current_cum_ratio += part_ratio

    return image_paths_per_group


# 각 그룹에 해당하는 이미지 경로의 이미지를 특정 디렉토리에 복사
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - image_paths_per_group (dict(list)) : 각 그룹에 해당하는 이미지의 리스트를 저장한 dict
# - dir_names_to_copy     (dict(str))  : 각 그룹에 해당하는 이미지를 복사할 경로
#                                        (key 조합은 image_paths_per_group 의 key 조합과 동일)

# Returns:
# - 각 그룹에 해당하는 이미지를 dir_names_to_copy 가 가리키는 디렉토리에 복사

def copy_images(image_paths_per_group, dir_names_to_copy):
    os.makedirs(f'{PROJECT_DIR_PATH}/{dir_names_to_copy}', exist_ok=True)

    raise NotImplementedError


# 원본 데이터 기준으로, 해당 카테고리의 test data 중 abnormal image 가 가장 많은 class 인 LAC 찾기
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - category_list (list(str)) : 세부 카테고리 목록

# Returns:
# - lac_dict (dict(str)) : 각 category 별 LAC 의 목록

def get_lac(category_list):
    lac_dict = {}

    for category in category_list:
        test_dataset_dir_name = f'{PROJECT_DIR_PATH}/mvtec_dataset_512/{category}/test'
        test_dataset_category_list = os.listdir(test_dataset_dir_name)
        num_imgs = []

        for test_dataset_category in test_dataset_category_list:
            if test_dataset_category == 'good':
                continue

            test_img_paths = os.listdir(f'{test_dataset_dir_name}/{test_dataset_category}')
            num_imgs.append([test_dataset_category, len(test_img_paths)])

        num_imgs.sort(key=lambda x: x[1], reverse=True)
        lac = num_imgs[0][0]
        lac_dict[category] = lac

    return lac_dict
