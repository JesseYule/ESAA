import numpy as np
import pandas as pd

import torch.optim as optim
import torch.nn as nn
import os
import torch
import torch.utils.data as Data

from network import net


# 这里预测和breast cancer相关的circRNA，breast cancer的index为9

# ------------训练集---------------------

circrna = np.loadtxt("test_circRNA_embedding.csv", delimiter=',')
circrna = np.mat(circrna)
print(circrna.shape)
# 一共有742个circRNA

disease = np.loadtxt("test_disease_embedding.csv", delimiter=',')
target = disease[34]
target = target[np.newaxis, :]

disease = np.repeat(target, 742, axis=0)
disease = np.mat(disease)

modelinput1 = torch.from_numpy(circrna)
modelinput2 = torch.from_numpy(disease)

modelinput1 = modelinput1.float()
modelinput2 = modelinput2.float()

model = net()

if os.path.exists('../checkpoint/model_back_95.pkl'):
    print('load model')
    model.load_state_dict(torch.load('../checkpoint/model_back_95.pkl'))

preds = model(modelinput1, modelinput2)

preds = preds.detach().numpy()

result_sheet = pd.read_excel('circRNA_index.xlsx', usecols=[0])
result_sheet.insert(1, 'prob', preds)


print(result_sheet)

result_sheet.to_csv("hcc.csv", index=False)

