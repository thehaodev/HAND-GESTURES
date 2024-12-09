import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from datetime import datetime
from torchmetrics import Accuracy
from torch.utils.data import Dataset

GESTURE_PATH = "hand_gesture.yaml"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file(GESTURE_PATH)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),  # Layer đầu tiên
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.Linear(128, 128),  # Layer thứ 2
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout với tỷ lệ 0.4
            nn.Linear(128, 128),  # Layer thứ 3
            nn.ReLU(),
            nn.Dropout(0.4),  # Dropout với tỷ lệ 0.4
            nn.Linear(128, 128),  # Layer thứ 4
            nn.ReLU(),
            nn.Dropout(0.6),  # Dropout với tỷ lệ 0.6
            nn.Linear(128, len(list_label))  # Output layer
        )

    def forward(self, x):
        x = self.flatten(x)  # Thực hiện flatten x
        x = self.linear_relu_stack(x)  # Pass x vào linear_relu_stack
        return x  # Return logits

    def predict(self, x, threshold=0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob, dim=1)
        return torch.where(softmax_prob[0, chosen_ind] > threshold, chosen_ind, -1)

    def predict_with_known_class(self, x):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob, dim=1)

    def score(self, logits):
        self()
        return -torch.amax(logits, dim=1)


class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.labels = torch.from_numpy(self.data.iloc[:, 0].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_hot_label = self.labels[idx]
        torch_data = torch.from_numpy(self.data.iloc[idx, 1:].to_numpy(dtype=np.float32))
        return torch_data, one_hot_label


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.watched_metrics = np.inf

    def early_stop(self, current_value):
        if current_value < self.watched_metrics:
            self.watched_metrics = current_value
            self.counter = 0
        elif current_value > (self.watched_metrics + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(train_loader, val_loader, model, loss_function, early_stopper, optimizer, list_label, save_path):
    # add auroc score
    acc_val = None
    best_model_path = None
    best_vloss = 1_000_000
    timestamp = datetime.now().strftime('%d-%m %H:%M')

    for epoch in range(300):
        # training step
        model.train(True)
        running_loss = 0.0
        acc_train = Accuracy(num_classes=len(list_label), task='multiclass')
        for batch_number, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_function(preds, labels)
            loss.backward()
            optimizer.step()
            acc_train.update(model.predict_with_known_class(inputs), labels)
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        # validating step
        model.train(False)
        running_vloss = 0.0
        acc_val = Accuracy(num_classes=len(list_label), task='multiclass')
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            preds = model(vinputs)
            vloss = loss_function(preds, vlabels)
            running_vloss += vloss.item()
            acc_val.update(model.predict_with_known_class(vinputs), vlabels)

        # Log the running loss averaged per batch
        # for both training and validation
        print(f"Epoch {epoch}: ")
        print(f"Accuracy train:{acc_train.compute().item()}, val:{acc_val.compute().item()}")
        avg_vloss = running_vloss / len(val_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('Training vs. Validation Loss',
              {'Training': avg_loss, 'Validation': avg_vloss},
              epoch + 1)
        print('Training vs. Validation accuracy',
              {'Training': acc_train.compute().item(), 'Validation': acc_val.compute().item()},
              epoch + 1)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_path = f'./{save_path}/model_{timestamp}_{model.__class__.__name__}_best'
            torch.save(model.state_dict(), best_model_path)

        if early_stopper.early_stop(avg_vloss):
            print(f" stopping at epoch {epoch}, minimum : {early_stopper.watched_metrics}")
            break

    model_path = f'./{save_path}/model_{timestamp}_{model.__class__.__name__}_last'
    torch.save(model.state_dict(), model_path)

    print(acc_val.compute())
    return model, best_model_path


def label_dict_from_config_file(relative_path):
    with open(relative_path, "r") as f:
        label_tag = yaml.full_load(f)["gestures"]
    return label_tag


def run():
    DATA_FOLDER_PATH = "./data/"
    LIST_LABEL = label_dict_from_config_file(GESTURE_PATH)
    train_path = os.path.join(DATA_FOLDER_PATH, "landmark_train.csv")
    val_path = os.path.join(DATA_FOLDER_PATH, "landmark_val.csv")
    save_path = './models'
    os.makedirs(save_path, exist_ok=True)

    train_set = CustomImageDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=40, shuffle=True, num_workers=0)

    val_set = CustomImageDataset(os.path.join(val_path))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=False, num_workers=0)

    model = NeuralNetwork()
    loss_function = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=30, min_delta=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

    model, best_model_path = train(train_loader, val_loader, model, loss_function,
                                   early_stopper, optimizer, LIST_LABEL, save_path)

    list_label = label_dict_from_config_file(GESTURE_PATH)
    DATA_FOLDER_PATH = "./data/"
    test_set = CustomImageDataset(os.path.join(DATA_FOLDER_PATH, "landmark_test.csv"))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=False, num_workers=0)

    network = NeuralNetwork()
    network.load_state_dict(torch.load(best_model_path, weights_only=True))

    network.eval()
    acc_test = Accuracy(num_classes=len(list_label), task='multiclass')
    for i, test_data in enumerate(test_loader):
        test_input, test_label = test_data
        preds = network(test_input)
        acc_test.update(preds, test_label)

    print(network.__class__.__name__)
    print(f"Accuracy of model:{acc_test.compute().item()}")
    print("========================================================================")


run()
