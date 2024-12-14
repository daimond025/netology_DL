import math
import random
from random import shuffle
import time
import pandas as pd
import torch


COUNT_TRAIN = 1000
COUNT_TEST = 100
ARRAY_SIZE = 5




def generation_data(len_data:int):
    cifre = list(range(0, 9))
    X, y = [], []
    for item in range(len_data):
        x_item = random.sample(list(range(0, 9)), ARRAY_SIZE)

        X.append(x_item)
        y_add = []
        for i,  x_ in enumerate(x_item):
            if i == 0:
                y_add.append(x_)
                continue

            y_ = x_ + x_item[0]

            if y_ >= 10:
                y_ -= 10
            y_add.append(y_)
        y.append(y_add)
    return torch.tensor(X), torch.tensor(y)


X_train, y_train = generation_data(COUNT_TRAIN)
X_test, y_test = generation_data(COUNT_TEST)


import torch.nn as nn

class ModelLST(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
class NeuralNetwork(nn.Module):
    def __init__(self, dictionary_size, embedding_size, num_hiddens):
        super().__init__()

        self.embedding = nn.Embedding(dictionary_size, embedding_size)
        self.hidden = nn.LSTM(embedding_size, num_hiddens, batch_first=True)
        self.output = nn.Linear(num_hiddens, 1)

    def forward(self, X):
        out = self.embedding(X)

        x, _ = self.hidden(out)
        predictions = self.output(x)
        return predictions

model = ModelLST(5, 50, 5  )

# model = NeuralNetwork(10, 30, 54  )
# y_pre = model(X_test[0:2].to(dtype=torch.float32))
# print(y_pre)
# exit()
# y_pre = y_pre.squeeze().flatten().tolist()
# y_pre = [ round(item,0) for item in y_pre]
# print(y_pre)
# exit()

import numpy as np
import torch.optim as optim
import torch.utils.data as data

optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

for epoch in range(500):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)

        y_batch = y_batch.flatten().to(dtype=torch.float32)
        y_pred = y_pred.squeeze().flatten().to(dtype=torch.float32)

        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 50 != 0:
        continue
    model.eval()

    with torch.no_grad():
        y_pred = model(X_train)
        y_pred = y_pred.squeeze().flatten().to(dtype=torch.float32)
        y_train_f = y_train.flatten().to(dtype=torch.float32)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train_f))

        y_pred = model(X_test)
        y_pred = y_pred.squeeze().flatten().to(dtype=torch.float32)
        y_test_f = y_test.flatten().to(dtype=torch.float32)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test_f))
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

x_pre = model(X_train[0])
print(x_pre)
print(x_pre.shape)

