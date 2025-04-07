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
from models.glass import get_model as get_glass_model
from models.tinyvit import get_model as get_tinyvit_model
from models.gradcam_original_code.utils.image import show_cam_on_image
from models.gradcam_original_code.utils.model_targets import ClassifierOutputTarget

try:
    from common_pytorch_training import run_train, run_validation, convert_anomaly_labels
except:
    from run_experiment.common_pytorch_training import run_train, run_validation, convert_anomaly_labels

import pandas as pd
import numpy as np
import PIL
import cv2
from enum import Enum
import random

import torch
import torch.nn as nn

import plotly.graph_objects as go


TRAIN_BATCH_SIZE_GLASS = 8
TRAIN_BATCH_SIZE_TINYVIT = 8  # to prevent CUDA OOM (with 12 GB GPU)
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
TINYVIT_EARLY_STOPPING_ROUNDS = 15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

anomaly_source_path = f'{PROJECT_DIR_PATH}/models/glass_anomaly_source'

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device for training : {device}')

# fix seeds (ref: https://github.com/cqylunlun/GLASS/blob/main/utils.py#L79)


def set_fixed_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    print(f'seed fixed as {seed}')


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

        # for GLASS test
        elif self.model_name == 'GLASS' and self.dataset_type == 'test':
            self.save_original_test_image(img_path, image)

            return image, label, img_path

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

    def save_original_test_image(self, img_path, image):
        category = img_path.split('/')[-4]

        if self.experiment_no == 1:
            dataset_path = f'{EXP1_GLASS_RESULT_PATH}/dataset_test/{category}'
        elif self.experiment_no == 2:
            dataset_path = f'{EXP2_GLASS_RESULT_PATH}/dataset_test/{category}'
        elif self.experiment_no == 3:
            dataset_path = f'{EXP3_GLASS_RESULT_PATH}/dataset_test/{category}'

        os.makedirs(dataset_path, exist_ok=True)
        image_save_path = f'{dataset_path}/{img_path.split("/")[-1]}'

        img_ = np.array(image)
        img_ = np.transpose(img_, (1, 2, 0)) * 255

        result, overlay_image_arr = cv2.imencode(ext='.png',
                                                 img=img_,
                                                 params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

        if result:
            with open(image_save_path, mode='w+b') as f:
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
# Last Update Date : 2025.04.04
# - NumPy, PyTorch Seed Fix (for reproducibility) 추가

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 GLASS 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)
# - category      (str)       : MVTec AD 데이터셋의 세부 카테고리 이름
# - experiment_no (int)       : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - entire_loss_list (list) : GLASS 모델의 Loss 기록

def run_train_glass(model, train_dataset, valid_dataset, category, experiment_no):
    set_fixed_seed(2025)

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
# Last Update Date : 2025.04.04
# - NumPy, PyTorch Seed Fix (for reproducibility) 추가
# - Early Stopping 을 Valid AUROC 기준으로 수정 및 이에 따라 Valid AUROC 반환값 추가

# Arguments:
# - model         (nn.Module) : 학습 및 성능 테스트에 사용할 TinyViT 모델
# - train_dataset (Dataset)   : 학습 데이터셋 (카테고리 별)
# - valid_dataset (Dataset)   : 검증 데이터셋 (카테고리 별)
# - category      (str)       : MVTec AD 데이터셋의 세부 카테고리 이름
# - experiment_no (int)       : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - val_accuracy_list (list) : Valid Accuracy 기록
# - val_loss_list     (list) : Valid Loss 기록
# - val_auroc_list    (list) : Valid AUROC 기록

def run_train_tinyvit(model, train_dataset, valid_dataset, category, experiment_no):
    set_fixed_seed(2025)

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
    val_accuracy_list, val_loss_list, val_auroc_list, best_epoch_model_with_softmax =(
        run_train_tinyvit_(model=model,
                           model_with_softmax=tinyvit_with_softmax,
                           train_loader=train_loader,
                           valid_loader=valid_loader))

    # save logs
    save_tinyvit_train_logs(val_accuracy_list, val_loss_list, val_auroc_list, experiment_no, category)

    # save trained TinyViT model
    model_path = f'{PROJECT_DIR_PATH}/run_experiment/exp{experiment_no}_tinyvit_ckpt/{category}'
    os.makedirs(model_path, exist_ok=True)

    best_epoch_no = np.argmax(val_accuracy_list)
    model_save_path = f'{model_path}/tinyvit_best_model_at_epoch_{best_epoch_no}.pt'
    torch.save(best_epoch_model_with_softmax.state_dict(), model_save_path)

    return val_accuracy_list, val_loss_list, val_auroc_list


# TinyViT 모델 학습 실시 (run_train_tinyvit 에서 호출)
# Create Date : 2025.04.03
# Last Update Date : 2025.04.04
# - Early Stopping 을 Valid AUROC 기준으로 수정 및 이에 따라 Valid AUROC 반환값 추가

# Arguments:
# - model              (nn.Module)  : 학습할 TinyViT 모델
# - model_with_softmax (nn.Module)  : Softmax 가 추가된 TinyViT 모델
# - train_loader       (DataLoader) : 학습 데이터셋 (카테고리 별) 의 DataLoader
# - valid_loader       (DataLoader) : 검증 데이터셋 (카테고리 별) 의 DataLoader

# Returns:
# - val_accuracy_list             (list)      : Valid Accuracy 기록
# - val_loss_list                 (list)      : Valid Loss 기록
# - val_auroc_list                (list)      : Valid AUROC 기록
# - best_epoch_model_with_softmax (nn.Module) : Softmax 가 추가된 TinyViT 모델 (at best epoch)

def run_train_tinyvit_(model, model_with_softmax, train_loader, valid_loader):
    current_epoch = 0
    max_val_auroc_epoch = -1  # AUROC-based Early Stopping
    max_val_auroc = None
    best_epoch_model_with_softmax = None

    val_accuracy_list = []
    val_loss_list = []
    val_auroc_list = []

    # loss function
    loss_func = nn.CrossEntropyLoss(reduction='sum')

    while True:
        train_loss = run_train(model=model_with_softmax,
                               train_loader=train_loader,
                               device=model_with_softmax.device,
                               loss_func=loss_func)

        val_accuracy, val_loss, val_auroc = run_validation(model=model_with_softmax,
                                                           valid_loader=valid_loader,
                                                           device=model_with_softmax.device,
                                                           loss_func=loss_func)

        print(f'epoch : {current_epoch}, train_loss : {train_loss:.4f}, '
              f'val_acc : {val_accuracy:.4f}, val_loss : {val_loss:.4f}, val_auroc : {val_auroc:.4f}')

        val_accuracy_list.append(val_accuracy)
        val_loss_cpu = float(val_loss.detach().cpu())
        val_loss_list.append(val_loss_cpu)
        val_auroc_list.append(val_auroc)

        model_with_softmax.scheduler.step()

        if max_val_auroc is None or val_auroc > max_val_auroc:
            max_val_auroc = val_auroc
            max_val_auroc_epoch = current_epoch

            best_epoch_model_with_softmax = TinyViTWithSoftmax(tinyvit_model=model,
                                                               num_classes=2).to(model_with_softmax.device)
            best_epoch_model_with_softmax.load_state_dict(model_with_softmax.state_dict())

        if current_epoch - max_val_auroc_epoch >= TINYVIT_EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return val_accuracy_list, val_loss_list, val_auroc_list, best_epoch_model_with_softmax


# TinyViT 모델 학습 결과 로그 저장
# Create Date : 2025.04.03
# Last Update Date : 2025.04.04
# - Early Stopping 기준이 Valid AUROC 로 변경됨에 따라 Valid AUROC 기록 추가

# Arguments:
# - val_accuracy_list (list) : valid accuracy 기록
# - val_loss_list     (list) : Valid Loss 기록
# - val_auroc_list    (list) : Valid AUROC 기록
# - experiment_no     (int)  : 실시할 실험 번호 (1, 2 또는 3)
# - category          (str)  : MVTec AD 데이터셋의 세부 카테고리 이름

# Returns:
# - log file 저장 (run_experiment/exp{1|2|3}_tinyvit_train_log/tinyvit_train_log.csv)

def save_tinyvit_train_logs(val_accuracy_list, val_loss_list, val_auroc_list, experiment_no, category):
    tinyvit_train_log_path = f'{PROJECT_DIR_PATH}/run_experiment/exp{experiment_no}_tinyvit_train_log/{category}'

    os.makedirs(tinyvit_train_log_path, exist_ok=True)
    train_log = {'max_val_acc': [], 'min_val_loss': [], 'max_val_auroc': [],
                 'total_epochs': [], 'best_epoch': [],
                 'val_acc_list': [], 'val_loss_list': [], 'val_auroc_list': []}

    max_val_acc = max(val_accuracy_list)
    max_val_acc_ = round(max_val_acc, 4)
    min_val_loss = min(val_loss_list)
    min_val_loss_ = round(min_val_loss, 4)
    max_val_auroc = max(val_auroc_list)
    max_val_auroc_ = round(max_val_auroc, 4)

    total_epochs = len(val_loss_list)
    val_accuracy_list_ = list(map(lambda x: round(x, 4), val_accuracy_list))
    val_loss_list_ = list(map(lambda x: round(x, 4), val_loss_list))
    val_auroc_list_ = list(map(lambda x: round(x, 4), val_auroc_list))

    train_log['max_val_acc'].append(max_val_acc_)
    train_log['min_val_loss'].append(min_val_loss_)
    train_log['max_val_auroc'].append(max_val_auroc_)

    train_log['total_epochs'].append(total_epochs)
    train_log['best_epoch'].append(np.argmax(val_accuracy_list_))

    train_log['val_acc_list'].append(val_accuracy_list_)
    train_log['val_loss_list'].append(val_loss_list_)
    train_log['val_auroc_list'].append(val_auroc_list_)

    train_log_path = f'{tinyvit_train_log_path}/tinyvit_train_log.csv'
    pd.DataFrame(train_log).to_csv(train_log_path)


# GLASS 모델 테스트 실시
# Create Date : 2025.04.03
# Last Update Date : 2025.04.04
# - score 및 label 정보 저장, 성능지표 계산 및 그 결과 저장을 별도 함수로 분리
# - NumPy, PyTorch Seed Fix (for reproducibility) 추가

# Arguments:
# - test_dataset  (Dataset)   : 테스트 데이터셋 (카테고리 별)
# - category      (str)       : MVTec AD 데이터셋의 세부 카테고리 이름
# - experiment_no (int)       : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def run_test_glass(test_dataset, category, experiment_no):
    set_fixed_seed(2025)

    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    exp_name = f'exp{experiment_no}'

    # define model
    model = get_glass_model()

    # load state dict
    model_dir = f'{PROJECT_DIR_PATH}/run_experiment/{exp_name}_glass_ckpt/exp1_anomaly_detection_{category}'
    model_file_name = list(filter(lambda x: x.endswith('_all.pth'), os.listdir(model_dir)))[0]
    model_path = f'{model_dir}/{model_file_name}'

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # run inference
    images, scores, segmentations, labels_gt, _, img_paths = model.predict(test_loader)

    # Overlay 이미지 저장
    exp_path = PROJECT_DIR_PATH + '/run_experiment/' + exp_name + '_glass_results'
    overlay_path = exp_path + '/overlay_test_data/' + exp_name + '_anomaly_detection_' + category
    os.makedirs(overlay_path, exist_ok=True)

    model.create_and_save_overlay_images(images, segmentations, img_paths, overlay_path)

    # score 및 label 정보 저장, 성능지표 계산 및 그 결과 저장
    thresholds = np.linspace(min(scores), max(scores), 500)

    test_result, confusion_matrix = save_score_label_metric_info(model_name='glass',
                                                                 exp_name=exp_name,
                                                                 category=category,
                                                                 thresholds=thresholds,
                                                                 img_paths=img_paths,
                                                                 scores=scores,
                                                                 labels_gt=labels_gt)

    return test_result, confusion_matrix


# TinyViT 학습된 모델 불러오기
# Create Date : 2025.04.04
# Last Update Date : -

# Arguments:
# - exp_name (str) : 실시할 실험 이름 ('exp1', 'exp2' 또는 'exp3')
# - category (str) : MVTec AD 데이터셋의 세부 카테고리 이름

# Returns:
# - model_with_softmax (nn.Module) : 학습된 TinyViT 모델 (마지막에 Softmax 적용된)

def load_tinyvit_trained_model(exp_name, category):

    # define model
    model = get_tinyvit_model()
    model_with_softmax = TinyViTWithSoftmax(model, num_classes=2)

    model_with_softmax.device = device
    model_with_softmax.to(device)

    # load state dict
    model_dir = f'{PROJECT_DIR_PATH}/run_experiment/{exp_name}_tinyvit_ckpt/{category}'
    model_file_name = list(filter(lambda x: x.endswith('.pt'), os.listdir(model_dir)))[0]
    model_path = f'{model_dir}/{model_file_name}'

    state_dict = torch.load(model_path, map_location=device)
    model_with_softmax.load_state_dict(state_dict, strict=True)

    return model_with_softmax


# TinyViT 모델 테스트 실시
# Create Date : 2025.04.03
# Last Update Date : 2025.04.04
# - TinyViT 학습된 모델을 불러오는 함수를 별도 함수로 분리

# Arguments:
# - test_dataset  (Dataset) : 테스트 데이터셋 (카테고리 별)
# - category      (str)     : MVTec AD 데이터셋의 세부 카테고리 이름
# - experiment_no (int)     : 실시할 실험 번호 (1, 2 또는 3)

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def run_test_tinyvit(test_dataset, category, experiment_no):
    set_fixed_seed(2025)

    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    exp_name = f'exp{experiment_no}'

    # get trained TinyViT model
    model_with_softmax = load_tinyvit_trained_model(exp_name, category)

    # save label, image path info and abnormal probability for test result files
    labels_gt = []
    img_paths = []
    probs = []

    for idx, (images_batch, labels_batch, img_paths_batch) in enumerate(test_loader):
        labels_gt += list(labels_batch)
        img_paths += list(img_paths_batch)

        labels_batch = convert_anomaly_labels(labels_batch)

        # run inference
        with torch.no_grad():
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device).to(torch.float32)
            probs_batch = model_with_softmax(images_batch)
            probs += list(probs_batch[:, 1].cpu().numpy())

    # score 및 label 정보 저장, 성능지표 계산 및 그 결과 저장
    thresholds = np.linspace(0.0, 1.0, 501)

    test_result, confusion_matrix = save_score_label_metric_info(model_name='tinyvit',
                                                                 exp_name=exp_name,
                                                                 category=category,
                                                                 thresholds=thresholds,
                                                                 img_paths=img_paths,
                                                                 scores=probs,
                                                                 labels_gt=labels_gt)

    return test_result, confusion_matrix


# 모델 테스트 실시 - score 및 label 정보 저장, 성능지표 계산 및 그 결과 저장
# Create Date : 2025.04.04
# Last Update Date : -

# Arguments:
# - model_name (str)      : 모델 이름 ('glass' 또는 'tinyvit')
# - exp_name   (str)      : 진행할 실험 이름 ('exp1', 'exp2' 또는 'exp3')
# - category   (str)      : MVTec AD 데이터셋의 세부 카테고리 이름
# - thresholds (np.array) : threshold list (anomaly score 의 min ~ max 범위)
# - img_paths  (list)     : 이미지 전체 경로의 리스트
# - scores     (list)     : anomaly score (또는 probability) 의 목록
# - labels_gt  (list)     : 이미지의 ground truth label 목록

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

# Materials to save:
# - test_result_df (threshold 별 성능지표 DataFrame 및 그 그래프), test_result, confusion_matrix 를 실험 결과 디렉토리에 저장
# - 각 sample 별 "경로, anomaly score, true label" 의 정보를 실험 결과 디렉토리에 저장

def save_score_label_metric_info(model_name, exp_name, category, thresholds, img_paths, scores, labels_gt):

    # 각 sample 별 score 및 label 정보 저장
    exp_path = PROJECT_DIR_PATH + '/run_experiment/' + exp_name + '_' + model_name + '_results'
    test_result_df_path = exp_path + '/test_result_df/' + exp_name + '_anomaly_detection_' + category
    os.makedirs(test_result_df_path, exist_ok=True)

    score_and_label_info_df = save_score_and_label_info(img_paths, scores, labels_gt)
    score_and_label_info_df.to_csv(f'{test_result_df_path}/score_and_label.csv', index=False)

    # 성능지표 계산 및 그 결과 저장
    test_result, test_result_df, confusion_matrix = compute_metric_values(thresholds, scores, labels_gt)

    test_result_df.to_csv(f'{test_result_df_path}/test_result_df.csv', index=False)
    save_test_result_df_as_chart(test_result_df, test_result_df_path)

    test_result_dict_df = pd.DataFrame({k: [v] for k, v in test_result.items()})
    test_result_dict_df.to_csv(f'{test_result_df_path}/test_result.csv', index=False)

    confusion_matrix.to_csv(f'{test_result_df_path}/confusion_matrix.csv', index=False)

    return test_result, confusion_matrix


# 모델 테스트 실시 - Metrics 값 계산
# Create Date : 2025.04.03
# Last Update Date : 2025.04.03
# - accuracy, recall 등 성능지표 값이 '-' 일 때 예외 처리 추가

# Arguments:
# - thresholds (np.array) : threshold list (anomaly score 의 min ~ max 범위)
# - scores     (list)     : anomaly score (또는 probability) 의 목록
# - labels_gt  (list)     : 이미지의 ground truth label 목록

# Returns:
# - test_result      (dict)             : 테스트 성능 평가 결과
#                                         {'accuracy': float, 'recall': float, 'precision': float, 'f1_score': float}
# - test_result_df   (Pandas DataFrame) : 각 threshold 별 성능지표 값을 저장한 DataFrame
# - confusion_matrix (Pandas DataFrame) : 테스트 성능 평가 시 생성된 Confusion Matrix

def compute_metric_values(thresholds, scores, labels_gt):
    thresholds_ = np.round(thresholds, 4)
    test_result_dict = {'threshold': thresholds_, 'accuracy': [], 'recall': [], 'precision': [], 'f1_score': []}
    best_f1_score = None

    for threshold in thresholds:
        tp, tn, fp, fn = 0, 0, 0, 0

        for score, label in zip(scores, labels_gt):
            if score >= threshold and label == 'abnormal':
                tp += 1
            elif score < threshold and label == 'normal':
                tn += 1
            elif score >= threshold and label == 'normal':
                fp += 1
            elif score < threshold and label == 'abnormal':
                fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = '-' if tp + fn == 0 else tp / (tp + fn)
        precision = '-' if tp + fp == 0 else tp / (tp + fp)

        f1_score_not_available = (recall == '-' or precision == '-' or recall + precision == 0)
        f1_score = '-' if f1_score_not_available else 2 * recall * precision / (recall + precision)

        test_result_dict['accuracy'].append('-' if accuracy == '-' else f'{accuracy:.4f}')
        test_result_dict['recall'].append('-' if recall == '-' else f'{recall:.4f}')
        test_result_dict['precision'].append('-' if precision == '-' else f'{precision:.4f}')
        test_result_dict['f1_score'].append('-' if f1_score == '-' else f'{f1_score:.4f}')

        # test result (tp,tn,fp,fn 개수 및 관련 성능지표 값들) 는 Best F1 Score 인 threshold 의 것을 이용
        best_record_updated = (f1_score != '-' and best_f1_score not in ['-', None] and f1_score > best_f1_score)

        if best_f1_score is None or (best_f1_score == '-' and f1_score != '-') or best_record_updated:
            best_f1_score = f1_score

            test_result = {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1_score': f1_score}

            confusion_matrix = pd.DataFrame(
                {'Pred \ True': ['Abnormal', 'Normal', 'Recall'],
                 'Abnormal'   : [tp, fn, f'{recall:.4f}'],
                 'Normal'     : [fp, tn, '-'],
                 'Precision'  : [f'{precision:.4f}', '-', f'acc: {accuracy:.4f}']}
            )

    test_result_df = pd.DataFrame(test_result_dict)

    return test_result, test_result_df, confusion_matrix


# 모델 테스트 실시 - 각 샘플 별 score 및 label 정보 저장
# Create Date : 2025.04.03
# Last Update Date : -

# Arguments:
# - img_paths (list) : 이미지 전체 경로의 리스트
# - scores    (list) : anomaly score (또는 probability) 의 목록
# - labels_gt (list) : 이미지의 ground truth label 목록

# Returns:
# - score_and_label_info_df (Pandas DataFrame) : 각 sample 별 "경로, anomaly score, true label" 의 정보

def save_score_and_label_info(img_paths, scores, labels_gt):
    score_and_label_info_dict = {'img_path': [], 'score': [], 'label': []}

    for img_path, score, label in zip(img_paths, scores, labels_gt):
        score_and_label_info_dict['img_path'].append(img_path)
        score_and_label_info_dict['score'].append(score)
        score_and_label_info_dict['label'].append(label)

    return pd.DataFrame(score_and_label_info_dict)


# 모델 테스트 실시 - threshold 별 성능 그래프를 실험 결과 디렉토리에 저장
# Create Date : 2025.04.03
# Last Update Date : 2025.04.03
# - accuracy, recall 등 성능지표 값이 '-' 일 때 예외 처리 추가

# Arguments:
# - test_result_df      (Pandas DataFrame) : 각 threshold 별 성능지표 값을 저장한 DataFrame
# - test_result_df_path (str)              : threshold 별 성능 그래프의 저장 경로

# Returns:
# - test_result_df 의 그래프를 실험 결과 디렉토리에 저장

def save_test_result_df_as_chart(test_result_df, test_result_df_path):
    fig = go.Figure()

    thresholds = test_result_df['threshold'].astype(float).tolist()
    accuracy = test_result_df['accuracy'].apply(lambda x: '-' if x == '-' else float(x)).tolist()
    recall = test_result_df['recall'].apply(lambda x: '-' if x == '-' else float(x)).tolist()
    precision = test_result_df['precision'].apply(lambda x: '-' if x == '-' else float(x)).tolist()
    f1_score = test_result_df['f1_score'].apply(lambda x: '-' if x == '-' else float(x)).tolist()

    metric_names = ['accuracy (%)', 'recall (%)', 'precision (%)', 'f1_score (%)']
    metric_values = [accuracy, recall, precision, f1_score]

    for metric_name, metric_value in zip(metric_names, metric_values):
        fig.add_trace(go.Scatter(x=thresholds,
                                 y=[('-' if m == '-' else 100.0 * m) for m in metric_value],
                                 mode='lines',
                                 name=metric_name))

    fig.update_layout(xaxis_title='anomaly threshold',
                      yaxis_title='metric value (%)')

    fig_path = f'{test_result_df_path}/test_result.png'
    fig.write_image(fig_path)  # need kaleido package (pip install kaleido)


# TinyViT 모델 설명 결과 출력 (ref: https://github.com/jacobgil/pytorch-grad-cam)
# Create Date : 2025.04.07
# Last Update Date : 2025.04.07
# - 레이어 이름 (layer_name) 을 인수로 추가 - 여러 레이어에 대한 XAI 결과 도출 목적
# - 오버레이 이미지 저장 디렉토리 이름 (overlay_img_dir_name) 을 인수로 추가 - Valid / Test Data 구분 대응 목적

# Arguments:
# - xai_model            (nn.Module)  : TinyViT 모델을 설명할 PyTorch Grad-CAM 모델
# - test_loader          (DataLoader) : Test Data 에 대한 Data Loader
# - category_name        (str)        : 카테고리 이름
# - layer_name           (str)        : XAI 결과를 도출할 레이어 이름 ('stage{M}_conv{N}' 형식)
# - experiment_no        (int)        : 실시할 실험 번호 (1, 2 또는 3)
# - overlay_img_dir_name (str)        : overlay image 를 저장할 디렉토리 이름 (기본값: 'overlay')

# Returns:
# - xai_output (PyTorch Tensor) : PyTorch Grad-CAM 모델의 출력

def run_tinyvit_explanation(xai_model, test_loader, category_name, layer_name, experiment_no,
                            overlay_img_dir_name='overlay'):

    targets = [ClassifierOutputTarget(1)]  # Abnormal Class No. = 1

    for idx, (images, labels, img_paths) in enumerate(test_loader):

        for image, img_path in zip(images, img_paths):
            image_ = image.unsqueeze(dim=0)
            grayscale_cam = xai_model(input_tensor=image_, targets=targets)[0]
            grayscale_cam_ = np.multiply(grayscale_cam, 255.0).astype(np.uint8)
            heatmap = cv2.applyColorMap(grayscale_cam_, cv2.COLORMAP_JET)

            img = np.array(image_[0])
            img = np.transpose(img, (1, 2, 0)) * 255

            img = img * IMAGENET_STD + IMAGENET_MEAN  # de-normalize
            overlay_image = 0.6 * img + 0.4 * heatmap

            # 이미지 저장 (이때 한글 경로 처리)
            overlay_path = f'{PROJECT_DIR_PATH}/run_experiment/exp{experiment_no}_tinyvit_results/{overlay_img_dir_name}'
            overlay_category_path = f'{overlay_path}/{layer_name}/{category_name}'
            overlay_save_path = f'{overlay_category_path}/{img_path.split("/")[-1]}'
            os.makedirs(overlay_category_path, exist_ok=True)

            result, overlay_image_arr = cv2.imencode(ext='.png',
                                                     img=overlay_image,
                                                     params=[cv2.IMWRITE_PNG_COMPRESSION, 0])

            if result:
                with open(overlay_save_path, mode='w+b') as f:
                    overlay_image_arr.tofile(f)
