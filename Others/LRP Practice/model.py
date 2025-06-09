
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary


INPUT_DIM = 6

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

EARLY_STOPPING_ROUNDS = 10
MAX_EPOCHS = 500


class ExampleNN(nn.Module):
    def __init__(self):
        super(ExampleNN, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Dropout(0.25)
        )
        self.fc_final = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_final(x)
        return x


class ExampleDataset(Dataset):
    def __init__(self, dataset_df):
        self.dataset_df = np.array(dataset_df)

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        return self.dataset_df[idx][:INPUT_DIM], self.dataset_df[idx][INPUT_DIM:]


def define_nn_model():
    nn = ExampleNN()
    nn.optimizer = torch.optim.AdamW(nn.parameters(), lr=0.01)
    nn.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=nn.optimizer, gamma=0.85)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    nn.to(device)
    nn.device = device

    # print summary of NN model before training
    summary(nn, input_size=(TRAIN_BATCH_SIZE, 6))

    return nn


def create_dataloader(example_dataset):
    dataset_size = len(example_dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = int(dataset_size * 0.1)
    test_size = dataset_size - (train_size + valid_size)

    train_dataset, valid_dataset, test_dataset = random_split(example_dataset, [train_size, valid_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, test_loader


def run_train(model, train_loader, device, loss_func=nn.MSELoss(reduction='sum')):
    model.train()

    for idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device).to(torch.float32), labels.to(device).to(torch.float32)

        # train 실시
        model.optimizer.zero_grad()
        outputs = model(data).to(torch.float32)

        loss = loss_func(outputs, labels)
        loss.backward()
        model.optimizer.step()


def run_validation(model, valid_loader, device, loss_func=nn.MSELoss(reduction='sum')):
    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device).to(torch.float32), labels.to(device).to(torch.float32)
            outputs = model(images).to(torch.float32)

            val_loss_batch = loss_func(outputs, labels)
            val_loss_sum += val_loss_batch
            total += labels.size(0)

        # Accuracy 계산
        val_accuracy = correct / total
        val_loss = val_loss_sum / total

    return val_accuracy, val_loss


def run_train_process(model, train_dataloader, valid_dataloader):
    current_epoch = 0
    min_val_loss_epoch = -1  # Loss-based Early Stopping
    min_val_loss = None
    best_epoch_model = None

    val_loss_list = []

    while True:
        run_train(model=model,
                  train_loader=train_dataloader,
                  device=model.device)

        _, val_loss = run_validation(model=model,
                                     valid_loader=valid_dataloader,
                                     device=model.device)

        print(f'epoch : {current_epoch}, val_loss : {val_loss:.6f}')
        val_loss_list.append(val_loss)

        model.scheduler.step()

        if min_val_loss is None or val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_epoch = current_epoch

            best_epoch_model = ExampleNN().to(model.device)
            best_epoch_model.load_state_dict(model.state_dict())

        if current_epoch >= MAX_EPOCHS or current_epoch - min_val_loss_epoch >= EARLY_STOPPING_ROUNDS:
            break

        current_epoch += 1

    return val_loss_list, best_epoch_model


def run_test_process(model, test_dataloader):
    true_scores = []
    pred_scores = []

    for idx, (data, labels) in enumerate(test_dataloader):
        with torch.no_grad():
            data, labels = data.to(model.device).to(torch.float32), labels.to(model.device).to(torch.float32)
            prediction = model(data).to(torch.float32).detach().cpu()

            true_scores += list(np.array(labels).flatten())
            pred_scores += list(np.array(prediction).flatten())

    mse = mean_squared_error(true_scores, pred_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = math.sqrt(mse)
    corrcoef = np.corrcoef(true_scores, pred_scores)[0][1]

    test_result_detail_dict = {'pred_score': pred_scores, 'true_score': true_scores}
    test_result_detail_df = pd.DataFrame(test_result_detail_dict)
    test_result_detail_df.to_csv('test_result_details.csv', index=False)

    performance_scores = {'mse': mse, 'mae': mae, 'rmse': rmse, 'corrcoef': corrcoef}
    return performance_scores


if __name__ == '__main__':
    example_nn = define_nn_model()

    example_dataset_df = pd.read_csv('dataset.csv')
    example_dataset = ExampleDataset(example_dataset_df)
    train_loader, valid_loader, test_loader = create_dataloader(example_dataset)

    _, best_epoch_model = run_train_process(example_nn, train_loader, valid_loader)
    performance_scores = run_test_process(example_nn, test_loader)

    print(performance_scores)
