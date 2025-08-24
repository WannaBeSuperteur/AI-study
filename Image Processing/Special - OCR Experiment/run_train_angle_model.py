
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.io import read_image

import numpy as np
import pandas as pd
import os


EARLY_STOPPING_ROUNDS = 10
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4


torch.set_printoptions(linewidth=160, sci_mode=False)
np.set_printoptions(suppress=True)

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])


class AngleModelDataset(Dataset):
    def __init__(self, dataset_df, transform):
        img_paths = dataset_df['img_path'].tolist()
        angle_labels = dataset_df['angle_0to1'].tolist()

        self.img_paths = img_paths
        self.angle_labels = angle_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        angle_label = self.angle_labels[idx]

        image = read_image(img_path)
        image = self.transform(image)  # normalize

        image_info = {'image': image, 'img_path': img_path}
        return image_info, angle_label


class ResNetAngleModel(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetAngleModel, self).__init__()
        self.resnet_model = resnet_model
        self.final_linear = nn.Linear(1000, 1)
        self.final_sigmoid = nn.Sigmoid(dim=1)

    def forward(self, x):
        x = self.resnet_model(x)
        x = self.final_linear(x)
        x = self.final_sigmoid(x)  # -15 degree to 0.0, +15 degree to +1.0, linearly
        return x


# Pre-train 된 ResNet18 모델 반환
# Create Date : 2025.08.24
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - angle_model (nn.module) : Pre-train 된 ResNet18 모델 기반 (Fine-Tuning 이전의) 회전 각도 예측 모델

def load_pretrained_model():
    pretrained_model = models.resnet18(pretrained=True)
    angle_model = ResNetAngleModel(resnet_model=pretrained_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_model.to(device)
    pretrained_model.device = device
    pretrained_model.optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=0.0001)

    return angle_model


# 모델 학습 실시
# Create Date : 2025.08.24
# Last Update Date : -

# args :
# - model        (nn.Module)  : 학습할 모델
# - train_loader (DataLoader) : Training Data Loader
# - device       (Device)     : CUDA or CPU device

# returns :
# - train_loss (float) : 모델의 Training Loss

def run_train(model, train_loader, device):
    model.train()
    total_images = 0
    train_loss_sum = 0.0

    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images['image'].to(device), labels.to(device).to(torch.float32)

        # train 실시
        model.optimizer.zero_grad()
        outputs = model(images).to(torch.float32)

        loss = nn.BCELoss(outputs, labels)
        loss.backward()
        model.optimizer.step()

        train_loss_sum += loss.item()
        total_images += labels.size(0)

    train_loss = train_loss_sum / total_images
    return train_loss


# 모델 validation / test 실시
# Create Date : 2025.08.24
# Last Update Date : -

# args :
# - model                (nn.Module)  : validation / test 할 모델
# - valid_or_test_loader (DataLoader) : Validation / Test Data Loader
# - device               (Device)     : CUDA or CPU device

# returns :
# - result (dict) : validation/test result (MSE error 및 Loss)

def run_valid_or_test(model, valid_or_test_loader, device):
    model.eval()

    total_images = 0
    mse_error_sum, loss_sum = 0.0, 0.0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_or_test_loader):
            images, labels = images['image'].to(device), labels.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)

            loss_batch = nn.BCELoss(outputs, labels)
            mse_error_batch = nn.MSELoss(outputs, labels)
            loss_sum += loss_batch
            mse_error_sum += mse_error_batch
            total_images += labels.size(0)

        # Loss 및 MSE Error 계산
        loss = loss_sum / total_images
        mse_error = mse_error_sum / total_images

    result = {'mse_error': mse_error, 'loss': loss}
    return result


# 모델 학습 프로세스 진행
# Create Date : 2025.08.24
# Last Update Date : -

# Arguments:
# - angle_model  (nn.Module)  : 학습할 ResNet 기반 회전 각도 예측 모델
# - train_loader (DataLoader) : 학습 Data Loader
# - valid_loader (DataLoader) : 검증 Data Loader

# Returns:
# - val_loss_list    (list)      : valid loss 기록
# - best_epoch_model (nn.Module) : valid loss 가 가장 낮은 epoch 에서의 모델

def run_train_process(model, train_loader, valid_loader):
    current_epoch = 0
    min_valid_loss_epoch = -1  # Loss-based Early Stopping
    min_valid_loss = None
    best_epoch_model = None

    val_loss_list = []

    while True:
        run_train(model=model,
                  train_loader=train_loader,
                  device=model.device)

        valid_result = run_valid_or_test(model=model,
                                         valid_or_test_loader=valid_loader,
                                         device=model.device)

        valid_mse_error, valid_loss = valid_result["mse_error"], valid_result["loss"]
        print(f'epoch : {current_epoch}, valid MSE error : {valid_mse_error:.6f}, valid loss : {valid_loss:.6f}')
        val_loss_list.append(valid_loss)

        if min_valid_loss is None or valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            min_valid_loss_epoch = current_epoch

            initial_angle_model = ResNetAngleModel(models.resnet18(pretrained=True))
            best_epoch_model = initial_angle_model.to(model.device)
            best_epoch_model.load_state_dict(model.state_dict())

        if current_epoch - min_valid_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return val_loss_list, best_epoch_model


# Document 회전 각도 예측 모델 전체 학습 실시
# Create Date : 2025.08.24
# Last Update Date : -

# Arguments:
# - train_loader (DataLoader) : Training Data Loader
# - valid_loader (DataLoader) : Valid Data Loader
# - test_loader  (DataLoader) : Test Data Loader

# Returns:
# - test_result (dict) : validation/test result (MSE error 및 Loss)

# Output:
# - valid loss 가 가장 낮은 epoch 의 Document 회전 각도 예측 모델을 저장

def run_model_pipeline(train_loader, valid_loader, test_loader):

    # train model
    initial_angle_model = load_pretrained_model()

    _, best_epoch_model = run_train_process(model=initial_angle_model,
                                            train_loader=train_loader,
                                            valid_loader=valid_loader)

    # test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_epoch_model.to(device)
    best_epoch_model.device = device

    test_result = run_valid_or_test(model=best_epoch_model,
                                    valid_or_test_loader=test_loader,
                                    device=device)

    # save model
    model_path = 'models/angle_model.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(best_epoch_model.state_dict(), model_path)

    print(f'test_result: {test_result}')
    return test_result


# Scanned Images Dataset 에 대한 DataLoader 생성
# Create Date : 2025.08.24
# Last Update Date : -

# Argument:
# - 없음

# Returns:
# - train_loader (DataLoader) : Training Data Loader
# - valid_loader (DataLoader) : Valid Data Loader
# - test_loader  (DataLoader) : Test Data Loader

def create_dataloader():

    # load DataFrame
    df_path = 'scanned_images_dataset/angle_label.csv'
    angle_label_df = pd.read_csv(df_path, index_col=0)
    angle_label_df['angle_0to1'] = angle_label_df['angle'].apply(lambda x: (x + 15.0) / 30.0)

    angle_label_df_train = angle_label_df[angle_label_df['img_path'].str.contains('train_rotated')]
    angle_label_df_test = angle_label_df[angle_label_df['img_path'].str.contains('test_rotated')]

    # DataFrame to dataset
    train_dataset = AngleModelDataset(dataset_df=angle_label_df_train, transform=image_transform)
    test_dataset = AngleModelDataset(dataset_df=angle_label_df_test, transform=image_transform)

    # Dataset to DataLoader
    train_size = int(len(train_dataset) * 0.8)
    valid_size = len(train_dataset) - train_size

    train_dataset_final, valid_dataset_final = random_split(train_dataset, [train_size, valid_size])
    test_dataset_final = test_dataset

    train_loader = DataLoader(train_dataset_final, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset_final, batch_size=VALID_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset_final, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = create_dataloader()
    run_model_pipeline(train_loader, valid_loader, test_loader)
