# 전체 데이터셋의 이미지 resize 실시
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - dest_size (int) : resize 이후의 이미지 가로/세로 길이 (default: 512)

# Returns:
# - mvtec_dataset 디렉토리 안의 모든 이미지를 dest_size x dest_size 로 resize

def resize_all_images(dest_size=512):
    raise NotImplementedError


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

    return NotImplementedError


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
    return NotImplementedError


# 원본 데이터 기준으로, 해당 카테고리의 test data 중 abnormal image 가 가장 많은 class 인 LAC 찾기
# Create Date : 2025.04.01
# Last Update Date : -

# Arguments:
# - category_list (list(str)) : 세부 카테고리 목록

# Returns:
# - lac_list (dict(str)) : 각 category 별 LAC 의 목록

def get_lac(category_list):
    return NotImplementedError
