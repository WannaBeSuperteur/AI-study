
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

LETTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']


torch.set_printoptions(linewidth=160, sci_mode=False)
np.set_printoptions(suppress=True)

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=0.5, std=0.5)  # -1.0 ~ +1.0 min-max normalization
])

loss_func = nn.CrossEntropyLoss(reduction='sum')


class LetterClassifyModelDataset(Dataset):
    def __init__(self, dataset_df, transform):
        img_paths = dataset_df['img_path'].tolist()
        letter_labels = dataset_df['letter_label'].tolist()

        self.img_paths = img_paths
        self.letter_labels = letter_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        letter_label = self.letter_labels[idx]
        letter_label_idx = LETTERS.index(letter_label)
        letter_label_one_hot = np.eye(len(LETTERS))[letter_label_idx]

        image = read_image(img_path)
        image = self.transform(image)  # normalize

        image_info = {'image': image, 'img_path': img_path}
        return image_info, letter_label_one_hot


class ResNetLetterClassifyModel(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetLetterClassifyModel, self).__init__()
        self.resnet_model = resnet_model
        self.final_linear = nn.Linear(1000, len(LETTERS))
        self.final_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.resnet_model(x)
        x = self.final_linear(x)
        x = self.final_softmax(x)
        return x


# Pre-train 된 ResNet18 모델 반환
# Create Date : 2025.08.26
# Last Update Date : -

# Arguments:
# - 없음

# Returns:
# - letter_classify_model (nn.module) : Pre-train 된 ResNet18 모델 기반 (Fine-Tuning 이전의) 글자 분류 모델

def load_pretrained_model():
    pretrained_model = models.resnet18(pretrained=True)
    pretrained_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    letter_classify_model = ResNetLetterClassifyModel(resnet_model=pretrained_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    letter_classify_model.to(device)
    letter_classify_model.device = device
    letter_classify_model.optimizer = torch.optim.AdamW(letter_classify_model.parameters(), lr=0.0001)
    letter_classify_model.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=letter_classify_model.optimizer,
                                                                             gamma=0.95)

    return letter_classify_model


# 모델 학습 실시
# Create Date : 2025.08.26
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
        loss = loss_func(outputs, labels)
        loss.backward()
        model.optimizer.step()

        train_loss_sum += loss.item()
        total_images += labels.size(0)

    train_loss = train_loss_sum / total_images
    return train_loss


# 모델 validation / test 실시
# Create Date : 2025.08.26
# Last Update Date : -

# args :
# - model                (nn.Module)  : validation / test 할 모델
# - valid_or_test_loader (DataLoader) : Validation / Test Data Loader
# - device               (Device)     : CUDA or CPU device

# returns :
# - result (dict) : validation/test result (accuracy 및 Loss)

def run_valid_or_test(model, valid_or_test_loader, device):
    model.eval()

    total_images, correct_images = 0, 0
    loss_sum = 0.0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_or_test_loader):
            images, labels = images['image'].to(device), labels.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)

            loss_batch = loss_func(outputs, labels)
            loss_sum += float(loss_batch.detach().cpu())

            # compute valid accuracy
            _, predicted = torch.max(outputs, 1)
            preds_cpu = list(np.array(outputs.detach().cpu()))
            labels_cpu = list(np.array(labels.detach().cpu()))

            for pred, label in zip(preds_cpu, labels_cpu):
                if np.argmax(pred) == np.argmax(label):
                    correct_images += 1

            total_images += labels.size(0)

        # Loss 계산
        loss = loss_sum / total_images

    accuracy = correct_images / total_images
    result = {'accuracy': accuracy, 'loss': loss}
    print(f'valid/test result : {result}')

    return result


# 글자 분류 모델 학습 프로세스 진행
# Create Date : 2025.08.26
# Last Update Date : -

# Arguments:
# - model        (nn.Module)  : 학습할 ResNet 기반 글자 분류 모델
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

        # run train & validation
        run_train(model=model,
                  train_loader=train_loader,
                  device=model.device)

        valid_result = run_valid_or_test(model=model,
                                         valid_or_test_loader=valid_loader,
                                         device=model.device)

        valid_accuracy, valid_loss = valid_result["accuracy"], valid_result["loss"]
        print(f'epoch : {current_epoch}, valid accuracy : {valid_accuracy:.6f}, valid loss : {valid_loss:.6f}')
        val_loss_list.append(valid_loss)

        # update scheduler
        model.scheduler.step()

        # handle early stopping
        if min_valid_loss is None or valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            min_valid_loss_epoch = current_epoch

            pretrained_model = models.resnet18(pretrained=True)
            pretrained_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            initial_angle_model = ResNetLetterClassifyModel(pretrained_model)

            best_epoch_model = initial_angle_model.to(model.device)
            best_epoch_model.load_state_dict(model.state_dict())

        if current_epoch - min_valid_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return val_loss_list, best_epoch_model


# 글자 분류 모델 전체 학습 실시
# Create Date : 2025.08.26
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
    initial_letter_classify_model = load_pretrained_model()

    _, best_epoch_model = run_train_process(model=initial_letter_classify_model,
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
    model_path = 'models/letter_classify_model.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(best_epoch_model.state_dict(), model_path)

    print(f'test_result: {test_result}')
    return test_result


# dataset info dictionary 에 이미지 경로 및 라벨 정보 추가
# Create Date : 2025.08.26
# Last Update Date : -

# Argument:
# - data_dir          (str)  : 데이터가 있는 디렉토리 경로
# - letter            (str)  : label 에 해당하는 글자 ('0', '1', '2', ...)
# - dataset_info_dict (dict) : dataset info dictionary

def add_image_path_and_label_info(data_dir: str, letter, dataset_info_dict):
    data_dir_letter = os.path.join(data_dir, letter)
    data_names = os.listdir(data_dir_letter)
    data_paths = [os.path.join(data_dir_letter, name) for name in data_names]

    dataset_info_dict['img_path'] += data_paths
    dataset_info_dict['letter_label'] += [letter] * len(data_names)


# 글자 예측 모델에 대한 DataLoader 생성
# Create Date : 2025.08.26
# Last Update Date : -

# Argument:
# - 없음

# Returns:
# - train_loader (DataLoader) : Training Data Loader
# - valid_loader (DataLoader) : Valid Data Loader
# - test_loader  (DataLoader) : Test Data Loader

def create_dataloader():
    train_dataset_info_dict = {'img_path': [], 'letter_label': []}
    test_dataset_info_dict = {'img_path': [], 'letter_label': []}

    train_data_dir = 'standard_ocr_dataset/data/training_data_modified'
    test_data_dir = 'standard_ocr_dataset/data/testing_data_modified'

    for letter in LETTERS:
        add_image_path_and_label_info(train_data_dir, letter, train_dataset_info_dict)
        add_image_path_and_label_info(test_data_dir, letter, test_dataset_info_dict)

    letter_class_df_train = pd.DataFrame(train_dataset_info_dict)
    letter_class_df_test = pd.DataFrame(test_dataset_info_dict)

    # DataFrame to dataset
    train_dataset = LetterClassifyModelDataset(dataset_df=letter_class_df_train, transform=image_transform)
    test_dataset = LetterClassifyModelDataset(dataset_df=letter_class_df_test, transform=image_transform)

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

