import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
EXP1_GLASS_RESULT_PATH = PROJECT_DIR_PATH + '/run_experiment/exp1_glass_results'

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

from models.glass_original_code.perlin import perlin_mask

import pandas as pd
import numpy as np
import torch
import PIL
import cv2
from enum import Enum

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

anomaly_source_path = f'{PROJECT_DIR_PATH}/models/glass_anomaly_source'


# ref : Original GLASS Implementation, from https://github.com/cqylunlun/GLASS/blob/main/datasets/mvtec.py

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class CustomMVTecDataset(Dataset):
    def __init__(self, dataset_df, transform, dataset_type, img_size, model_name):
        self.img_paths = dataset_df['img_path'].tolist()
        self.labels = dataset_df['label'].tolist()
        self.transform = transform
        self.distribution = 2  # (for GLASS, Anomaly Detection) suppose MANIFOLD distribution for all sub-categories
        self.dataset_type = dataset_type
        self.img_size = img_size
        self.model_name = model_name

        # for GLASS
        if model_name == 'GLASS':
            self.mean = 0.5
            self.std = 0.1
            self.rand_aug = 1
            self.downsampling = 8

            self.transform_img = [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img = transforms.Compose(self.transform_img)
            self.anomaly_source_paths = self.get_anomaly_source_paths()

    def get_anomaly_source_paths(self):
        anomaly_source_dir = f'{PROJECT_DIR_PATH}/models/glass_anomaly_source'
        anomaly_source_categories = os.listdir(anomaly_source_dir)

        anomaly_source_paths = []

        for category in anomaly_source_categories:
            category_dir_path = f'{anomaly_source_dir}/{category}'
            anomaly_source_imgs = os.listdir(category_dir_path)
            anomaly_source_imgs = [f'{category_dir_path}/{x}' for x in anomaly_source_imgs]

            anomaly_source_paths += anomaly_source_imgs

        return anomaly_source_paths

    def __len__(self):
        return len(self.img_paths)

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.img_size),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

    def get_aug_image_and_mask_s(self, img_path):
        image = PIL.Image.open(img_path).convert("RGB")
        image = self.transform_img(image)

        mask_fg = mask_s = aug_image = torch.tensor([1])

        aug = PIL.Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
        transform_aug = self.rand_augmenter()
        aug = transform_aug(aug)

        mask_all = perlin_mask(image.shape, self.img_size // self.downsampling, 0, 6, mask_fg, 1)
        mask_s = torch.from_numpy(mask_all[0])
        mask_l = torch.from_numpy(mask_all[1])

        beta = np.random.normal(loc=self.mean, scale=self.std)
        beta = np.clip(beta, .2, .8)
        aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

        return aug_image, mask_s

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        image = self.transform(image)
        label = self.labels[idx]

        # for GLASS training
        if self.model_name == 'GLASS' and self.dataset_type == 'train':
            aug_image, mask_s = self.get_aug_image_and_mask_s(img_path)
            self.save_original_and_aug_image(img_path, image, aug_image)

            return image, label, img_path, aug_image, mask_s
        else:
            return image, label, img_path

    def save_original_and_aug_image(self, img_path, image, aug_image):
        category = img_path.split('/')[-4]
        dataset_path = f'{EXP1_GLASS_RESULT_PATH}/dataset/{category}'
        os.makedirs(dataset_path, exist_ok=True)

        image_save_path = f'{dataset_path}/{img_path.split("/")[-1]}'
        aug_image_save_path = f'{dataset_path}/{img_path.split("/")[-1].split(".")[0]}_aug.png'

        # 이미지 저장 시 한글 경로 처리
        for img, save_path in zip([image, aug_image], [image_save_path, aug_image_save_path]):
            img_ = np.array(img)
            img_ = np.transpose(img_, (1, 2, 0)) * 255

            result, overlay_image_arr = cv2.imencode(ext='.png',
                                                     img=img_,
                                                     params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

            if result:
                with open(save_path, mode='w+b') as f:
                    overlay_image_arr.tofile(f)


# 학습, 검증 및 테스트 데이터셋 정의
# Create Date : 2025.04.02
# Last Update Date : 2025.04.03
# - img_size 변수 추가 (Dataset Class 에서 사용)
# - model_name 변수 추가

# Arguments:
# - category_name    (str) : 카테고리 이름
# - dataset_dir_name (str) : 데이터셋 디렉토리 이름
# - img_size         (int) : 이미지 크기 (256 또는 512)
# - model_name       (str) : 모델 이름 ('TinyViT' or 'GLASS')

# Returns:
# - train_dataset (Dataset) : 해당 카테고리의 학습 데이터셋
# - valid_dataset (Dataset) : 해당 카테고리의 검증 데이터셋
# - test_dataset  (Dataset) : 해당 카테고리의 테스트 데이터셋

def get_datasets(category_name, dataset_dir_name, img_size, model_name):
    train_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='train')
    valid_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='valid')
    test_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='test')

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])

    train_dataset = CustomMVTecDataset(train_dataset_df,
                                       transform,
                                       dataset_type='train',
                                       img_size=img_size,
                                       model_name=model_name)

    valid_dataset = CustomMVTecDataset(valid_dataset_df,
                                       transform,
                                       dataset_type='valid',
                                       img_size=img_size,
                                       model_name=model_name)

    test_dataset = CustomMVTecDataset(test_dataset_df,
                                      transform,
                                      dataset_type='test',
                                      img_size=img_size,
                                      model_name=model_name)

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
# Last Update Date : 2025.04.03
# - category 별 checkpoint 구분을 위해 category name 인수 추가

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)
# - category      (str)       : MVTec AD 데이터셋의 세부 카테고리 이름

# Returns:
# - entire_loss_list (list) : GLASS 모델의 Loss 기록

def run_train_glass(model, train_dataset, valid_dataset, category):
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    glass_model_dir = f'{PROJECT_DIR_PATH}/run_experiment/exp1_glass_ckpt'
    model.set_model_dir(glass_model_dir, dataset_name=f"exp1_anomaly_detection_{category}")

    entire_loss_list = model.trainer(train_loader, valid_loader, name=f"exp1_anomaly_detection_{category}")
    return entire_loss_list


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
# Create Date : 2025.04.03
# Last Update Date : -

# Arguments:
# - model        (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - test_dataset (Dataset)   : 테스트 데이터셋 (카테고리 별)
# - category     (str)       : MVTec AD 데이터셋의 세부 카테고리 이름

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def run_test_glass(model, test_dataset, category):
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    _, _, _, _, _, _, anomaly_score_info = model.tester(test_loader, name=f"exp1_anomaly_detection_{category}")

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
