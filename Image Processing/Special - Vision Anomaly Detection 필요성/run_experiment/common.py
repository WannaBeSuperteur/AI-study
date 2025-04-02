import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

import pandas as pd


class CustomMVTecDataset(Dataset):
    def __init__(self, dataset_df, transform):
        self.img_paths = dataset_df['img_path'].tolist()
        self.labels = dataset_df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = f'{PROJECT_DIR_PATH}/{self.img_paths[idx]}'
        image = read_image(img_path)
        image = self.transform(image)
        label = self.labels[idx]

        return image, label


# 학습, 검증 및 테스트 데이터셋 정의
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - category_name    (str) : 카테고리 이름
# - dataset_dir_name (str) : 데이터셋 디렉토리 이름

# Returns:
# - train_dataset (Dataset) : 해당 카테고리의 학습 데이터셋
# - valid_dataset (Dataset) : 해당 카테고리의 검증 데이터셋
# - test_dataset  (Dataset) : 해당 카테고리의 테스트 데이터셋

def get_datasets(category_name, dataset_dir_name):
    train_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='train')
    valid_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='valid')
    test_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='test')

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    train_dataset = CustomMVTecDataset(train_dataset_df, transform)
    valid_dataset = CustomMVTecDataset(valid_dataset_df, transform)
    test_dataset = CustomMVTecDataset(test_dataset_df, transform)

    return train_dataset, valid_dataset, test_dataset


# 학습, 검증 및 테스트 데이터셋을 정의하기 위한 Pandas DataFrame 생성
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - category_name    (str) : 카테고리 이름
# - dataset_dir_name (str) : 데이터셋 디렉토리 이름
# - dataset_type     (str) : 'train', 'valid' or 'test'

# Returns:
# - dataset_df (Pandas DataFrame) : 데이터셋 정보가 저장된 Pandas DataFrame
#                                   columns = ['img_path', 'label']

def create_dataset_df(category_name, dataset_dir_name, dataset_type):
    dataset_dir_path = f'{PROJECT_DIR_PATH}/{dataset_dir_name}/{category_name}/{dataset_type}'

    img_paths = []
    labels = []

    # 이미지 찾기
    for (path, _, files) in os.walk(dataset_dir_path):
        path_ = path.replace(os.sep, '/')

        for file_name in files:
            ext = os.path.splitext(file_name)[-1]

            if ext == '.png':
                img_path = f'{path_}/{file_name}'
                label = path_.split('/')[-1]

                img_paths.append(img_path)
                labels.append(label)

    dataset_dict = {'img_path': img_paths, 'label': labels}
    dataset_df = pd.DataFrame(dataset_dict)

    return dataset_df


# GLASS 모델 학습 실시
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)

# Returns:
# - val_loss_list (list) : Valid Loss 기록

def run_train_glass(model, train_dataset, valid_dataset):
    raise NotImplementedError


# TinyViT 모델 학습 실시
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 TinyViT 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)

# Returns:
# - val_loss_list (list) : Valid Loss 기록

def run_train_tinyvit(model, train_dataset, valid_dataset):
    raise NotImplementedError


# GLASS 모델 테스트 실시
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - model        (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - test_dataset (Dataset)   : 테스트 데이터셋 (카테고리 별)

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def run_test_glass(model, test_dataset):
    raise NotImplementedError


# TinyViT 모델 테스트 실시
# Create Date : 2025.04.02
# Last Update Date : -

# Arguments:
# - model        (nn.Module) : 학습 및 성능 테스트에 사용할 TinyViT 모델
# - test_dataset (Dataset)   : 테스트 데이터셋 (카테고리 별)

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def run_test_tinyvit(model, test_dataset):
    raise NotImplementedError
