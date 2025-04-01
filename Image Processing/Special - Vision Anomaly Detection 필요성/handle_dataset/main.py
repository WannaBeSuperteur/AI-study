from tvt_split import resize_all_images, split_dataset_into_groups, copy_images, get_lac

import os


# MVTec-AD Dataset 의 카테고리 목록
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - category_list (list) : MVTec-AD 데이터셋의 전체 카테고리 목록

def get_category_list():
    file_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    mvtec_dataset_path = f'{file_path}/mvtec_dataset'

    return os.listdir(mvtec_dataset_path)


# 정량적 성능 평가 - Vision Anomaly Detection 의 Train/Valid/Test 데이터 구분
# Create Date : 2025.04.01
# Last Update Date : -

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
    image_path_train = split_dataset_into_groups(dataset_type='train',
                                                 dataset_class='normal',
                                                 category_list=category_list,
                                                 group_names=['train_normal'],
                                                 split_ratio=[1.0])

    image_path_test_normal = split_dataset_into_groups(dataset_type='test',
                                                       dataset_class='normal',
                                                       category_list=category_list,
                                                       group_names=['valid_normal', 'test_normal'],
                                                       split_ratio=[0.5, 0.5])

    image_path_test_abnormal = split_dataset_into_groups(dataset_type='test',
                                                         dataset_class='abnormal',
                                                         category_list=category_list,
                                                         group_names=['valid_abnormal', 'test_abnormal'],
                                                         split_ratio=[0.5, 0.5])

    image_path_dicts = [image_path_train, image_path_test_normal, image_path_test_abnormal]

    for image_path_dict in image_path_dicts:
        copy_images(image_paths_per_group=image_path_dict,
                    dir_names_to_copy='mvtec_dataset_exp1_anomaly')


# 정량적 성능 평가 - Vision Classification 의 Train/Valid/Test 데이터 구분
# Create Date : 2025.04.01
# Last Update Date : -

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
    image_path_train = split_dataset_into_groups(dataset_type='train',
                                                 dataset_class='normal',
                                                 category_list=category_list,
                                                 group_names=['train_normal'],
                                                 split_ratio=[1.0])

    image_path_test_normal = split_dataset_into_groups(dataset_type='test',
                                                       dataset_class='normal',
                                                       category_list=category_list,
                                                       group_names=['valid_normal', 'test_normal'],
                                                       split_ratio=[0.5, 0.5])

    image_path_test_abnormal = split_dataset_into_groups(dataset_type='test',
                                                         dataset_class='abnormal',
                                                         category_list=category_list,
                                                         group_names=['train_abnormal', 'valid_abnormal', 'test_abnormal'],
                                                         split_ratio=[0.5, 0.25, 0.25])

    image_path_dicts = [image_path_train, image_path_test_normal, image_path_test_abnormal]

    for image_path_dict in image_path_dicts:
        copy_images(image_paths_per_group=image_path_dict,
                    dir_names_to_copy='mvtec_dataset_exp1_classify')


# 새로운 Abnormal Class 탐지 성능 평가 - Vision Classification 의 Train/Valid/Test 데이터 구분
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - category_list (list(str)) : 세부 카테고리 목록
# - lac_list      (dict(str)) : 각 category 별 LAC 의 목록

# Returns:
# - mvtec_dataset_exp3_classify/{category_name} 디렉토리에,
# - 각 카테고리 별,
#   - 원본 데이터셋의 Train Data            -> Train Data 의 Normal Sample 로 분리
#   - 원본 데이터셋의 Test Data  (Normal)   -> Valid Data 의 Normal Sample 로 분리
#   - 원본 데이터셋의 Test Data  (Abnormal) -> Train Data 의 Abnormal Sample (LAC 의 75%) +
#                                           Valid Data 의 Abnormal Sample (LAC 의 25%) +
#                                           Test Data 의 Abnormal Sample (LAC 외의 모든 Class) 로 분리

def split_data_exp3_classify(category_list, lac_list):
    image_path_train = split_dataset_into_groups(dataset_type='train',
                                                 dataset_class='normal',
                                                 category_list=category_list,
                                                 group_names=['train_normal'],
                                                 split_ratio=[1.0])

    image_path_test_normal = split_dataset_into_groups(dataset_type='test',
                                                       dataset_class='normal',
                                                       category_list=category_list,
                                                       group_names=['valid_normal'],
                                                       split_ratio=[1.0])

    image_path_test_abnormal = None  # TODO: implement

    image_path_dicts = [image_path_train, image_path_test_normal, image_path_test_abnormal]

    for image_path_dict in image_path_dicts:
        copy_images(image_paths_per_group=image_path_dict,
                    dir_names_to_copy='mvtec_dataset_exp3_classify')

    raise NotImplementedError


if __name__ == '__main__':

    # get category list
    category_list = get_category_list()
    lac_list = get_lac(category_list)

    print(f'CATEGORY LIST:\n{category_list}')
    print(f'LAC LIST:\n{lac_list}')

    resize_all_images()

    # experiment dataset setting
    split_data_exp1_anomaly(category_list)
    split_data_exp1_classify(category_list)
    split_data_exp3_classify(category_list, lac_list)
