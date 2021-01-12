import numpy as np
import pandas as pd

import torch.optim as optim
import torch.nn as nn
import os
import torch
import torch.utils.data as Data

from network import net

BATCH_SIZE = 1000
learning_rate = 1e-3
epochs = 10000

# 模型输入，其实就是独热编码
input = np.eye(742)
input = torch.from_numpy(input)

output = np.loadtxt("adjacency_matrix.csv", delimiter=',')
output = np.mat(output)
output = torch.from_numpy(output)

torch_dataset = Data.TensorDataset(input, output)

loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=2,              # 多线程来读数据
)


model = net()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss(reduction='mean')


if os.path.exists('checkpoint/model.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/model.pkl'))


for epoch in range(epochs):

    train_acc = 0
    train_loss = 0

    for step, (modelinput, modeloutput) in enumerate(loader):

        modelinput = modelinput.float()
        modeloutput = modeloutput.float()

        optimizer.zero_grad()

        preds, embedding_vector = model(modelinput)

        train_loss = loss_fn(preds, modeloutput)
        train_loss += train_loss.item()

        train_loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print('epoch ', epoch, ': ', 'train loss: ', train_loss.item())

    if epoch % 100 == 0:
        # print('save model')
        torch.save(model.state_dict(), 'checkpoint/model.pkl')
