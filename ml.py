import random
import sys
from dvclive import Live
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
sns.set()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

data = pd.read_csv('data/iris1.csv')
train_test_data = [data]
for dataset in train_test_data:
    dataset['variety'] = dataset['variety'].map( 'A {}'.format )
for dataset in train_test_data:
    dataset['variety'] = dataset['variety'].map( {"A Setosa": 0, "A Versicolor": 1, "A Virginica": 2} )

X = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = data[['variety']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


def get_dataset(X, y):
    tensor_x = torch.Tensor(X.values)
    tensor_y = torch.Tensor(y.values).long()

    dataset = TensorDataset(tensor_x,tensor_y)
    return dataset


train_dataset = get_dataset(X_train, y_train)
test_dataset = get_dataset(X_test, y_test)

params = {"epochs": 15}

def train(net, train_loader, device, num_epochs, learning_rate):

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()
    acc_history = []

    with Live(save_dvc_exp=True) as live:
        
        for param in params:
            live.log_param(param, params[param])
        
        for epoch in range(1, params["epochs"]):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_num, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = net(inputs)
                labels = labels.view(1, 10)
                labels = labels[0]
                loss = loss_function(outputs, labels)

                # Backpropagation
                loss.backward()

                # Update
                optimizer.step()

                # Print progress
                running_loss += loss.item()

                # Calculate batch Accuracy
                _, predicted = outputs.max(1)
                batch_total = labels.size(0)
                batch_correct = predicted.eq(labels).sum().item()
                batch_acc = batch_correct/batch_total

                total += batch_total
                correct += batch_correct

            # Print the evaluation metric and reset it for the next epoch
            acc = correct/total
            acc_history.append(acc)

            live.log_metric("train/accuracy", acc)
            live.log_metric("train/loss", running_loss)
            #live.log_metric("val/accuracy", epoch + random.random() )
            #live.log_metric("val/loss", epochs - epoch - random.random())
            live.next_step()
    
    return acc_history

def print_history(history, title):
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

BATCH_SIZE = 10
EPOCHS = 10
LR = 0.01

train_dataloader = DataLoader(train_dataset, shuffle= True, batch_size = BATCH_SIZE)

class Net(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classifier(x)
        return x
    
Net = Net()
hist_Net = train(Net, train_dataloader, device, EPOCHS, LR)
print_history(hist_Net, "Net Model Accuracy")

def evaluate_acc(net, test_loader, device):

    total = 0
    correct = 0

    for batch_num, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = correct/total
    return acc

test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
Net_acc = evaluate_acc(Net, test_dataloader, device)