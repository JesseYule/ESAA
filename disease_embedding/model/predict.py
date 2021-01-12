import numpy as np
import os
import torch

from network import net

import pandas as pd


# 模型输入，其实就是独热编码
input = np.eye(743)

# input1就是circRNA的独热编码
test_input1 = torch.from_numpy(input)

# 因为采用siamese结构，所以把输入变成模型的另一个输入，每行表示一个miRNA对应的输出向量
result1 = np.loadtxt("adjacency_matrix.csv", delimiter=',')
result1 = np.mat(result1)
result1 = 1 - result1
test_input2 = torch.from_numpy(result1)

model = net()

if os.path.exists('checkpoint/model.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/model.pkl'))

modelinput1 = test_input1.float()
modelinput2 = test_input2.float()

preds, embedding_vector = model(modelinput1, modelinput2)

preds = preds.detach().numpy()

print(preds)

preds[preds < 0.5] = 0
preds[preds >= 0.5] = 1
correct_num = np.sum(preds == 0)

print(correct_num)


