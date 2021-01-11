import numpy as np
import os
import torch

from network import net

input = np.eye(742)
input = torch.from_numpy(input)

model = net()

if os.path.exists('checkpoint/model.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/model.pkl'))

modelinput = input.float()

preds, embedding_vector = model(modelinput)

embedding_vector = embedding_vector.detach().numpy()

np.savetxt("circRNA_embedding.csv", embedding_vector, delimiter=',')




