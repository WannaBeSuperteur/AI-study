import torch.nn as nn
import torch
import numpy as np  # for test code
from sklearn import metrics

is_test = False

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)


# Anomaly Detection 데이터셋을 학습할 때, 그 데이터셋의 label 을 abnormal -> 1, normal -> 0 으로 변환
# Create Date : 2025.04.03
# Last Update Date : -

# args :
# - labels (tuple) : 각 batch 의 이미지들의 label 리스트

# returns :
# - torch_labels (Tensor) : PyTorch Tensor 형태의 0과 1로 변환된 label 리스트

def convert_anomaly_labels(labels):
    labels_list = [1 if label == 'abnormal' else 0 for label in labels]
    return torch.tensor(labels_list)


# 모델 학습 실시
# Create Date : 2025.04.03
# Last Update Date : -

# args :
# - model        (nn.Module)  : 학습할 모델
# - train_loader (DataLoader) : Training Data Loader
# - device       (Device)     : CUDA or CPU device
# - loss_func    (func)       : Loss Function

# returns :
# - train_loss (float) : 모델의 Training Loss

def run_train(model, train_loader, device, loss_func=nn.CrossEntropyLoss(reduction='sum')):
    model.train()
    total = 0
    train_loss_sum = 0.0

    for idx, (images, labels, img_paths) in enumerate(train_loader):
        labels = convert_anomaly_labels(labels)
        images, labels = images.to(device), labels.to(device).to(torch.float32)

        # train 실시
        model.optimizer.zero_grad()
        outputs = model(images).to(torch.float32)[:, 1]

        loss = loss_func(outputs, labels)
        loss.backward()
        model.optimizer.step()

        # test code
        if is_test and idx % 5 == 0:
            print('train idx:', idx)
            print('output:', np.array(outputs.detach().cpu()).flatten())
            print('label:', np.array(labels.detach().cpu()))

        train_loss_sum += loss.item()
        total += labels.size(0)

    train_loss = train_loss_sum / total
    return train_loss


# 모델 validation 실시
# Create Date : 2025.04.03
# Last Update Date : 2025.04.04
# - AUROC 성능지표 계산 및 반환값으로 추가

# args :
# - model        (nn.Module)  : validation 할 모델
# - valid_loader (DataLoader) : Validation Data Loader
# - device       (Device)     : CUDA or CPU device
# - loss_func    (func)       : Loss Function

# returns :
# - val_accuracy (float) : 모델의 validation 정확도
# - val_loss     (float) : 모델의 validation loss
# - val_auroc    (float) : 모델의 validation AUROC 값

def run_validation(model, valid_loader, device, loss_func=nn.CrossEntropyLoss(reduction='sum')):
    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0

    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for idx, (images, labels, img_paths) in enumerate(valid_loader):
            labels = convert_anomaly_labels(labels)
            images, labels = images.to(device), labels.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)

            val_loss_batch = loss_func(outputs[:, 1], labels)
            val_loss_sum += val_loss_batch

            all_labels += list(labels.cpu().numpy())
            all_outputs += list(outputs[:, 1].cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # test code
            if is_test and idx % 5 == 0:
                print('valid idx:', idx)
                print('output:', np.array(outputs[:, 1].detach().cpu()))
                print('label:', np.array(labels.detach().cpu()))

        # Accuracy 계산
        val_accuracy = correct / total
        val_loss = val_loss_sum / total

    val_auroc = metrics.roc_auc_score(all_labels, all_outputs)

    return val_accuracy, val_loss, val_auroc

