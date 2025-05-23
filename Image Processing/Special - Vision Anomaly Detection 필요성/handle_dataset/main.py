try:
    from tvt_split import (resize_all_images,
                           split_dataset_into_groups,
                           split_dataset_into_groups_lac,
                           copy_images,
                           get_lac)
except:
    from handle_dataset.tvt_split import (resize_all_images,
                                          split_dataset_into_groups,
                                          split_dataset_into_groups_lac,
                                          copy_images,
                                          get_lac)

import numpy as np
import cv2

import os
PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

TINYVIT_IMG_SIZE = 512
GLASS_IMG_SIZE = 256  # to reduce GPU memory (12GB)


# MVTec-AD Dataset 의 카테고리 목록
# Create Date : 2025.04.01
# Last Update Date : 2025.04.02
# - *.txt file 제외 처리
# - MVTec dataset 디렉토리 이름 변경 반영

# Arguments:
# - 없음

# Returns:
# - category_list (list) : MVTec-AD 데이터셋의 전체 카테고리 목록

def get_category_list():
    mvtec_dataset_path = f'{PROJECT_DIR_PATH}/mvtec_dataset_512'

    file_list = os.listdir(mvtec_dataset_path)
    category_list = list(filter(lambda x: '.' not in x, file_list))

    return category_list


# 정량적 성능 평가 - Vision Anomaly Detection 의 Train/Valid/Test 데이터 구분
# Create Date : 2025.04.01
# Last Update Date : 2025.04.02
# - 변수명 수정 : dir_names_to_copy -> dir_name_to_copy
# - split_dataset_into_groups 함수에 original_data_dir 인수 추가 반영

# Arguments:
# - category_list (list(str)) : 세부 카테고리 목록

# Returns:
# - mvtec_dataset_exp1_anomaly/{category_name} 디렉토리에,
# - 각 카테고리 별,
#   - 원본 데이터셋의 Train Data            -> Train Data 의 Normal Sample 로 분리
#   - 원본 데이터셋의 Test Data  (Normal)   -> Valid Data 의 Normal Sample (50%) +
#                                           Test Data 의 Normal Sample (50%) 로 분리
#   - 원본 데이터셋의 Test Data  (Abnormal) -> Valid Data 의 Abnormal Sample (50%) +
#                                           Test Data 의 Abnormal Sample (50%) 로 분리

def split_data_exp1_anomaly(category_list):
    original_data_dir = 'mvtec_dataset_256'

    image_path_train = split_dataset_into_groups(dataset_type='train',
                                                 dataset_class='normal',
                                                 category_list=category_list,
                                                 group_names=['train_normal'],
                                                 split_ratio=[1.0],
                                                 original_data_dir=original_data_dir)

    image_path_test_normal = split_dataset_into_groups(dataset_type='test',
                                                       dataset_class='normal',
                                                       category_list=category_list,
                                                       group_names=['valid_normal', 'test_normal'],
                                                       split_ratio=[0.5, 0.5],
                                                       original_data_dir=original_data_dir)

    image_path_test_abnormal = split_dataset_into_groups(dataset_type='test',
                                                         dataset_class='abnormal',
                                                         category_list=category_list,
                                                         group_names=['valid_abnormal', 'test_abnormal'],
                                                         split_ratio=[0.5, 0.5],
                                                         original_data_dir=original_data_dir)

    image_path_dicts = [image_path_train, image_path_test_normal, image_path_test_abnormal]

    for image_path_dict in image_path_dicts:
        copy_images(image_paths_per_group=image_path_dict,
                    dir_name_to_copy='mvtec_dataset_exp1_anomaly')


# 정량적 성능 평가 - Vision Classification 의 Train/Valid/Test 데이터 구분
# Create Date : 2025.04.01
# Last Update Date : 2025.04.02
# - 변수명 수정 : dir_names_to_copy -> dir_name_to_copy
# - split_dataset_into_groups 함수에 original_data_dir 인수 추가 반영

# Arguments:
# - category_list (list(str)) : 세부 카테고리 목록

# Returns:
# - mvtec_dataset_exp1_classify/{category_name} 디렉토리에,
# - 각 카테고리 별,
#   - 원본 데이터셋의 Train Data            -> Train Data 의 Normal Sample 로 분리
#   - 원본 데이터셋의 Test Data  (Normal)   -> Valid Data 의 Normal Sample (50%) +
#                                           Test Data 의 Normal Sample (50%) 로 분리
#   - 원본 데이터셋의 Test Data  (Abnormal) -> Train Data 의 Abnormal Sample (50%) +
#                                           Valid Data 의 Abnormal Sample (25%) +
#                                           Test Data 의 Abnormal Sample (25%) 로 분리

def split_data_exp1_classify(category_list):
    original_data_dir = 'mvtec_dataset_512'

    image_path_train = split_dataset_into_groups(dataset_type='train',
                                                 dataset_class='normal',
                                                 category_list=category_list,
                                                 group_names=['train_normal'],
                                                 split_ratio=[1.0],
                                                 original_data_dir=original_data_dir)

    image_path_test_normal = split_dataset_into_groups(dataset_type='test',
                                                       dataset_class='normal',
                                                       category_list=category_list,
                                                       group_names=['valid_normal', 'test_normal'],
                                                       split_ratio=[0.5, 0.5],
                                                       original_data_dir=original_data_dir)

    image_path_test_abnormal = split_dataset_into_groups(dataset_type='test',
                                                         dataset_class='abnormal',
                                                         category_list=category_list,
                                                         group_names=['train_abnormal', 'valid_abnormal', 'test_abnormal'],
                                                         split_ratio=[0.5, 0.25, 0.25],
                                                         original_data_dir=original_data_dir)

    image_path_dicts = [image_path_train, image_path_test_normal, image_path_test_abnormal]

    for image_path_dict in image_path_dicts:
        copy_images(image_paths_per_group=image_path_dict,
                    dir_name_to_copy='mvtec_dataset_exp1_classify')


# 새로운 Abnormal Class 탐지 성능 평가 - Vision Classification 의 Train/Valid/Test 데이터 구분
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - category_list (list(str)) : 세부 카테고리 목록
# - lac_dict      (dict(str)) : 각 category 별 LAC 의 목록

# Returns:
# - mvtec_dataset_exp3_classify/{category_name} 디렉토리에,
# - 각 카테고리 별,
#   - 원본 데이터셋의 Train Data            -> Train Data 의 Normal Sample 로 분리
#   - 원본 데이터셋의 Test Data  (Normal)   -> Valid Data 의 Normal Sample 로 분리
#   - 원본 데이터셋의 Test Data  (Abnormal) -> Train Data 의 Abnormal Sample (LAC 의 75%) +
#                                           Valid Data 의 Abnormal Sample (LAC 의 25%) +
#                                           Test Data 의 Abnormal Sample (LAC 외의 모든 Class) 로 분리

def split_data_exp3_classify(category_list, lac_dict):
    original_data_dir = 'mvtec_dataset_512'

    image_path_train = split_dataset_into_groups(dataset_type='train',
                                                 dataset_class='normal',
                                                 category_list=category_list,
                                                 group_names=['train_normal'],
                                                 split_ratio=[1.0],
                                                 original_data_dir=original_data_dir)

    image_path_test_normal = split_dataset_into_groups(dataset_type='test',
                                                       dataset_class='normal',
                                                       category_list=category_list,
                                                       group_names=['valid_normal'],
                                                       split_ratio=[1.0],
                                                       original_data_dir=original_data_dir)

    image_path_test_abnormal = split_dataset_into_groups_lac(dataset_type='test',
                                                             category_list=category_list,
                                                             lac_group_names=['train_abnormal', 'valid_abnormal'],
                                                             lac_split_ratio=[0.75, 0.25],
                                                             others_group_names=['test_abnormal'],
                                                             others_split_ratio=[1.0],
                                                             original_data_dir=original_data_dir,
                                                             lac_dict=lac_dict)

    image_path_dicts = [image_path_train, image_path_test_normal, image_path_test_abnormal]

    for image_path_dict in image_path_dicts:
        copy_images(image_paths_per_group=image_path_dict,
                    dir_name_to_copy='mvtec_dataset_exp3_classify')


# 90,180,270도 회전 Data Augmentation 실시 (Training Abnormal Data / Classification task only)
# Create Date : 2025.04.03
# Last Update Date : -

# Arguments:
# - dir_name      (str)       : Augmentation 을 실시할 데이터셋 디렉토리 이름
# - category_list (list(str)) : 세부 카테고리 목록

# Returns:
# - 데이터셋 디렉토리의 모든 sub-dataset 에 대해 Train Abnormal Data 가 90,180,270도 회전 Augmentation 으로 4배 증대

def do_augmentation(dir_name, category_list):
    dataset_dir_path = f'{PROJECT_DIR_PATH}/{dir_name}'
    train_abnormal_dirs = [f'{dataset_dir_path}/{category}/train/abnormal' for category in category_list]

    for train_abnormal_dir in train_abnormal_dirs:
        train_abnormal_imgs = os.listdir(train_abnormal_dir)
        train_abnormal_imgs = list(filter(lambda x: x.endswith('.png'), train_abnormal_imgs))
        train_abnormal_img_paths = [f'{train_abnormal_dir}/{img_name}' for img_name in train_abnormal_imgs]

        for path in train_abnormal_img_paths:
            augment_and_save_img(path)


# 각 이미지에 대해 90,180,270도 회전 Data Augmentation 실시
# Create Date : 2025.04.03
# Last Update Date : -

# Arguments:
# - path (str) : 이미지 전체 경로

# Returns:
# - 해당 이미지에 대해 Train Abnormal Data 가 90,180,270도 회전 Augmentation 으로 4배 증대

def augment_and_save_img(path):

    # 이미지 읽기 시 한글 경로 처리
    train_abnormal_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    train_abnormal_img_90 = cv2.rotate(train_abnormal_img, cv2.ROTATE_90_CLOCKWISE)
    train_abnormal_img_180 = cv2.rotate(train_abnormal_img, cv2.ROTATE_180)
    train_abnormal_img_270 = cv2.rotate(train_abnormal_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    augmented_imgs = [train_abnormal_img_90, train_abnormal_img_180, train_abnormal_img_270]
    angles = [90, 180, 270]

    for augemented_img, angle in zip(augmented_imgs, angles):

        # 이미지 저장 시 한글 경로 처리
        result, img_arr = cv2.imencode(ext='.png', img=augemented_img, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
        augemented_img_save_path = path[:-4] + '_rotate_' + str(angle) + '.png'

        if result:
            with open(augemented_img_save_path, mode='w+b') as f:
                img_arr.tofile(f)


if __name__ == '__main__':

    # get category list
    category_list = get_category_list()
    lac_dict = get_lac(category_list)

    print(f'CATEGORY LIST:\n{category_list}')
    print(f'LAC DICT:\n{lac_dict}')

    # execute ONLY AT FIRST TIME resizing image
    resize_all_images(img_dir='mvtec_dataset_512', dest_size=TINYVIT_IMG_SIZE)
    resize_all_images(img_dir='mvtec_dataset_256', dest_size=GLASS_IMG_SIZE)

    # experiment dataset setting
    split_data_exp1_anomaly(category_list)
    split_data_exp1_classify(category_list)
    split_data_exp3_classify(category_list, lac_dict)

    # data augmentation for training abnormal data
    do_augmentation('mvtec_dataset_exp1_classify', category_list)
    do_augmentation('mvtec_dataset_exp3_classify', category_list)

