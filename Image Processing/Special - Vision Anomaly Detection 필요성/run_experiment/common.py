import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

PROJECT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
EXP1_GLASS_RESULT_PATH = PROJECT_DIR_PATH + '/run_experiment/exp1_glass_results'
EXP2_GLASS_RESULT_PATH = PROJECT_DIR_PATH + '/run_experiment/exp2_glass_results'
EXP3_GLASS_RESULT_PATH = PROJECT_DIR_PATH + '/run_experiment/exp3_glass_results'


from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

from models.glass_original_code.perlin import perlin_mask

try:
    from common_pytorch_training import run_train, run_validation
except:
    from run_experiment.common_pytorch_training import run_train, run_validation

import pandas as pd
import numpy as np
import PIL
import cv2
from enum import Enum

import torch
import torch.nn as nn

TRAIN_BATCH_SIZE_GLASS = 16
TRAIN_BATCH_SIZE_TINYVIT = 8  # to prevent CUDA OOM (with 12 GB GPU)
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
TINYVIT_EARLY_STOPPING_ROUNDS = 7

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

anomaly_source_path = f'{PROJECT_DIR_PATH}/models/glass_anomaly_source'

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device for training : {device}')


# ref : Original GLASS Implementation, from https://github.com/cqylunlun/GLASS/blob/main/datasets/mvtec.py

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class CustomMVTecDataset(Dataset):
    def __init__(self, dataset_df, transform, dataset_type, img_size, model_name, experiment_no):
        self.img_paths = dataset_df['img_path'].tolist()
        self.labels = dataset_df['label'].tolist()
        self.transform = transform
        self.distribution = 2  # (for GLASS, Anomaly Detection) suppose MANIFOLD distribution for all sub-categories
        self.dataset_type = dataset_type
        self.img_size = img_size
        self.model_name = model_name
        self.experiment_no = experiment_no  # 1, 2 or 3 (for exp1, exp2 and exp3, respectively)

        assert experiment_no in [1, 2, 3], "EXPERIMENT NO MUST BE IN [1, 2, 3] for each experiment."

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

        if self.experiment_no == 1:
            dataset_path = f'{EXP1_GLASS_RESULT_PATH}/dataset/{category}'
        elif self.experiment_no == 2:
            dataset_path = f'{EXP2_GLASS_RESULT_PATH}/dataset/{category}'
        elif self.experiment_no == 3:
            dataset_path = f'{EXP3_GLASS_RESULT_PATH}/dataset/{category}'

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


class TinyViTWithSoftmax(nn.Module):
    def __init__(self, tinyvit_model, num_classes):
        super(TinyViTWithSoftmax, self).__init__()

        self.tinyvit_model = tinyvit_model
        self.num_classes = num_classes

        self.final_softmax = nn.Softmax()

    def forward(self, x):
        x = self.tinyvit_model(x)
        x = self.final_softmax(x)
        return x


# 학습, 검증 및 테스트 데이터셋 정의
# Create Date : 2025.04.02
# Last Update Date : 2025.04.03
# - 원본 이미지를 ImageNet Normalize 하는 부분의 누락 해결
# - experiment_no (실험의 번호) 인수 추가

# Arguments:
# - category_name    (str) : 카테고리 이름
# - dataset_dir_name (str) : 데이터셋 디렉토리 이름
# - img_size         (int) : 이미지 크기 (256 또는 512)
# - model_name       (str) : 모델 이름 ('TinyViT' or 'GLASS')
# - experiment_no    (int) : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - train_dataset (Dataset) : 해당 카테고리의 학습 데이터셋
# - valid_dataset (Dataset) : 해당 카테고리의 검증 데이터셋
# - test_dataset  (Dataset) : 해당 카테고리의 테스트 데이터셋

def get_datasets(category_name, dataset_dir_name, img_size, model_name, experiment_no):
    train_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='train')
    valid_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='valid')
    test_dataset_df = create_dataset_df(category_name, dataset_dir_name, dataset_type='test')

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    train_dataset = CustomMVTecDataset(train_dataset_df,
                                       transform,
                                       dataset_type='train',
                                       img_size=img_size,
                                       model_name=model_name,
                                       experiment_no=experiment_no)

    valid_dataset = CustomMVTecDataset(valid_dataset_df,
                                       transform,
                                       dataset_type='valid',
                                       img_size=img_size,
                                       model_name=model_name,
                                       experiment_no=experiment_no)

    test_dataset = CustomMVTecDataset(test_dataset_df,
                                      transform,
                                      dataset_type='test',
                                      img_size=img_size,
                                      model_name=model_name,
                                      experiment_no=experiment_no)

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
# - batch size 상수 수정
# - experiment_no (실험의 번호) 인수 추가

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)
# - category      (str)       : MVTec AD 데이터셋의 세부 카테고리 이름
# - experiment_no (int)       : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - entire_loss_list (list) : GLASS 모델의 Loss 기록

def run_train_glass(model, train_dataset, valid_dataset, category, experiment_no):
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE_GLASS, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    exp_name = f'exp{experiment_no}'

    glass_model_dir = f'{PROJECT_DIR_PATH}/run_experiment/{exp_name}_glass_ckpt'
    model.experiment_no = experiment_no
    model.set_model_dir(glass_model_dir, dataset_name=f"{exp_name}_anomaly_detection_{category}")

    entire_loss_list = model.trainer(train_loader, valid_loader, name=f"{exp_name}_anomaly_detection_{category}")
    return entire_loss_list


# TinyViT 모델 학습 실시
# Create Date : 2025.04.03
# Last Update Date : 2025.04.03
# - experiment_no (실험의 번호) 인수 추가

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 TinyViT 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)
# - experiment_no (int)       : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - val_loss_list (list) : Valid Loss 기록

def run_train_tinyvit(model, train_dataset, valid_dataset, experiment_no):
    print(f'train dataset size : {len(train_dataset)}')
    print(f'valid dataset size : {len(valid_dataset)}')

    # define TinyViT
    tinyvit_with_softmax = TinyViTWithSoftmax(model, num_classes=2)
    tinyvit_with_softmax.optimizer = torch.optim.AdamW(tinyvit_with_softmax.parameters(), lr=0.001)
    tinyvit_with_softmax.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=tinyvit_with_softmax.optimizer,
                                                                                T_max=10,
                                                                                eta_min=0)
    tinyvit_with_softmax.device = device

    # map to CUDA GPU
    model.to(device)
    tinyvit_with_softmax.to(device)

    # define DataLoader
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE_TINYVIT, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    # train TinyViT
    val_loss_list, best_epoch_model_with_softmax = run_train_tinyvit_(model=model,
                                                                      model_with_softmax=tinyvit_with_softmax,
                                                                      train_loader=train_loader,
                                                                      valid_loader=valid_loader)

    # save logs
    save_tinyvit_train_logs(val_loss_list, experiment_no)

    # save trained TinyViT model
    model_path = f'{PROJECT_DIR_PATH}/run_experiment/exp{experiment_no}_tinyvit_ckpt'
    os.makedirs(model_path, exist_ok=True)

    model_save_path = f'{model_path}/tinyvit_trained_model.pt'
    torch.save(best_epoch_model_with_softmax.state_dict(), model_save_path)

    return val_loss_list


# TinyViT 모델 학습 실시 (run_train_tinyvit 에서 호출)
# Create Date : 2025.04.03
# Last Update Date : -

# Arguments:
# - model              (nn.Module)  : 학습할 TinyViT 모델
# - model_with_softmax (nn.Module)  : Softmax 가 추가된 TinyViT 모델
# - train_loader       (DataLoader) : 학습 데이터셋 (카테고리 별) 의 DataLoader
# - valid_loader       (DataLoader) : 검증 데이터셋 (카테고리 별) 의 DataLoader

# Returns:
# - val_loss_list                 (list)      : Valid Loss 기록
# - best_epoch_model_with_softmax (nn.Module) : Softmax 가 추가된 TinyViT 모델 (at best epoch)

def run_train_tinyvit_(model, model_with_softmax, train_loader, valid_loader):
    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model_with_softmax = None

    val_loss_list = []

    # loss function
    loss_func = nn.CrossEntropyLoss(reduction='sum')

    while True:
        train_loss = run_train(model=model_with_softmax,
                               train_loader=train_loader,
                               device=model_with_softmax.device,
                               loss_func=loss_func)

        val_accuracy, val_loss = run_validation(model=model_with_softmax,
                                                valid_loader=valid_loader,
                                                device=model_with_softmax.device,
                                                loss_func=loss_func)

        print(f'epoch : {current_epoch}, '
              f'train_loss : {train_loss:.4f}, val_acc : {val_accuracy:.4f}, val_loss : {val_loss:.4f}')

        val_loss_cpu = float(val_loss.detach().cpu())
        val_loss_list.append(val_loss_cpu)

        model_with_softmax.scheduler.step()

        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = current_epoch

            best_epoch_model_with_softmax = TinyViTWithSoftmax(tinyvit_model=model,
                                                               num_classes=2).to(model_with_softmax.device)
            best_epoch_model_with_softmax.load_state_dict(model_with_softmax.state_dict())

        if current_epoch - min_val_loss_epoch >= TINYVIT_EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return val_loss_list, best_epoch_model_with_softmax


# TinyViT 모델 학습 결과 로그 저장
# Create Date : 2025.04.03
# Last Update Date : -

# Arguments:
# - val_loss_list (list) : Valid Loss 기록
# - experiment_no (int)  : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - log file 저장 (run_experiment/exp{1|2|3}_tinyvit_train_log/tinyvit_train_log.csv)

def save_tinyvit_train_logs(val_loss_list, experiment_no):
    tinyvit_train_log_path = f'{PROJECT_DIR_PATH}/run_experiment/exp{experiment_no}_tinyvit_train_log'

    os.makedirs(tinyvit_train_log_path, exist_ok=True)
    train_log = {'min_val_loss': [], 'total_epochs': [], 'best_epoch': [], 'val_loss_list': []}

    min_val_loss = min(val_loss_list)
    min_val_loss_ = round(min_val_loss, 4)
    total_epochs = len(val_loss_list)
    val_loss_list_ = list(map(lambda x: round(x, 4), val_loss_list))

    train_log['min_val_loss'].append(min_val_loss_)
    train_log['total_epochs'].append(total_epochs)
    train_log['best_epoch'].append(np.argmin(val_loss_list_))
    train_log['val_loss_list'].append(val_loss_list_)

    train_log_path = f'{tinyvit_train_log_path}/tinyvit_train_log.csv'
    pd.DataFrame(train_log).to_csv(train_log_path)


# GLASS 모델 테스트 실시
# Create Date : 2025.04.03
# Last Update Date : -

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - test_dataset  (Dataset)   : 테스트 데이터셋 (카테고리 별)
# - category      (str)       : MVTec AD 데이터셋의 세부 카테고리 이름
# - experiment_no (int)       : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def run_test_glass(model, test_dataset, category, experiment_no):
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    _, _, _, _, _, _, anomaly_score_info = model.tester(test_loader,
                                                        name=f"exp{experiment_no}_anomaly_detection_{category}")

    raise NotImplementedError


# TinyViT 모델 테스트 실시
# Create Date : 2025.04.03
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
