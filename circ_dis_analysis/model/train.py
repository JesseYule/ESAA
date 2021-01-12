import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


import torch.optim as optim
import torch.nn as nn
import os
import torch
import torch.utils.data as Data

from network import net


BATCH_SIZE = 4000
learning_rate = 1e-5
epochs = 20
threshold = 0.5

print_loss_frequency = 10
print_train_accuracy_frequency = 100
test_frequency = 10

mini_loss = 0.5
maxauc = 0.92

show_test_detail = False
plot_loss = False

Loss_list = []
Accuracy_list = []

# ------------训练集---------------------

positive_input1 = np.loadtxt("positive_circRNA_vector.csv", delimiter=',')
positive_input1 = np.mat(positive_input1)
positive_input2 = np.loadtxt("positive_disease_vector.csv", delimiter=',')
positive_input2 = np.mat(positive_input2)

negative_input1 = np.loadtxt("negative_circRNA_vector.csv", delimiter=',')
negative_input1 = np.mat(negative_input1)
negative_input2 = np.loadtxt("negative_disease_vector.csv", delimiter=',')
negative_input2 = np.mat(negative_input2)

#  从数据集中抽部分数据作为测试集，然后删除数据集中的这部分数据
# 正负样本各846条数据，做5折交叉检验

selected_index = np.arange(4, 846, 5)  # 抽五分一数据出来做检查检验，正负样本各169条

# 筛选出测试集
test_positive_input1 = positive_input1[selected_index]
test_positive_input2 = positive_input2[selected_index]

# 从训练集中删除测试集数据
positive_input1 = np.delete(positive_input1, selected_index, axis=0)
positive_input2 = np.delete(positive_input2, selected_index, axis=0)

test_negative_input1 = negative_input1[selected_index]
test_negative_input2 = negative_input2[selected_index]

negative_input1 = np.delete(negative_input1, selected_index, axis=0)
negative_input2 = np.delete(negative_input2, selected_index, axis=0)

input1 = np.concatenate((positive_input1, negative_input1), axis=0)
input2 = np.concatenate((positive_input2, negative_input2), axis=0)

input1 = torch.from_numpy(input1)
input2 = torch.from_numpy(input2)
output1 = np.ones(positive_input1.shape[0])
output2 = np.zeros(negative_input1.shape[0])
output = np.concatenate((output1, output2), axis=0)
output = torch.from_numpy(output)

train_torch_dataset = Data.TensorDataset(input1, input2, output)

train_loader = Data.DataLoader(
    dataset=train_torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=2,              # 多线程来读数据
)


# ----------测试集------------------

test_input1 = np.concatenate((test_positive_input1, test_negative_input1), axis=0)
test_input2 = np.concatenate((test_positive_input2, test_negative_input2), axis=0)
test_input1 = torch.from_numpy(test_input1)
test_input2 = torch.from_numpy(test_input2)
test_output1 = np.ones(test_positive_input1.shape[0])
test_output2 = np.zeros(test_negative_input1.shape[0])
test_output = np.concatenate((test_output1, test_output2), axis=0)
test_output = torch.from_numpy(test_output)

test_torch_dataset = Data.TensorDataset(test_input1, test_input2, test_output)

test_loader = Data.DataLoader(
    dataset=test_torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    # num_workers=2,              # 多线程来读数据
)

model = net()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss(reduction='mean')


if os.path.exists('checkpoint/new_model_4.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/new_model_4.pkl'))


for epoch in range(epochs):

    train_acc = 0
    train_loss = 0

    for step, (train_input1, train_input2, train_output) in enumerate(train_loader):

        # print('训练集数据： ', len(train_input1))

        train_input1 = train_input1.float()
        train_input2 = train_input2.float()

        train_output = train_output.float()

        optimizer.zero_grad()

        train_preds = model(train_input1, train_input2)

        train_loss = loss_fn(train_preds, train_output)
        train_loss += train_loss.item()

        train_loss.backward()
        optimizer.step()

    if epoch % print_loss_frequency == 0:
        # print('------------  epoch ', epoch, '  ------------')
        print('train loss: ', train_loss.item())

    if epoch % test_frequency == 0:
        for step, (test_input1, test_input2, test_output) in enumerate(test_loader):

            # print('测试集数据： ', len(test_input1))

            test_input1 = test_input1.float()
            test_input2 = test_input2.float()

            test_output = test_output.float()

            test_preds = model(test_input1, test_input2)

            test_preds = test_preds.detach().numpy()
            test_output = test_output.detach().numpy()

            test_preds_origin = test_preds

            auc_value = roc_auc_score(test_output, test_preds)
            print('AUC: ', auc_value)

            # 以下内容用于输出模型的输出和真实值，用于画ROC曲线
            # if epoch == 10:
            #     np.savetxt('output4.txt', test_output)
            #     np.savetxt('predict4.txt', test_preds_origin)


            test_preds[test_preds < threshold] = 0
            test_preds[test_preds >= threshold] = 1

            tp = 0
            tn = 0

            for i in range(len(test_preds)):
                if test_preds[i] == test_output[i] and test_preds[i] == 1:
                    tp += 1

            for i in range(len(test_preds)):
                if test_preds[i] == test_output[i] and test_preds[i] == 0:
                    tn += 1

            # positive、negative的数量都等于正负测试样本的数量，所以算出tp、tn的数量后直接用正负测试样本数量相减即可
            fp = len(selected_index) - tp
            fn = len(selected_index) - tn

            print('TP: ', tp)
            print('FN: ', fn)
            print('TN: ', tn)
            print('FP: ', fp)

            if tp+fn == 0:
                sen = 0
            else:
                sen = tp/(tp+fn)

            prec = tp/(tp+fp)
            f1 = 2*tp/(2*tp+fp+fn)

            correct_num = np.sum(test_preds == test_output)
            test_accuracy = correct_num/len(test_preds)

            print(' correct num: ', correct_num, ' whole num: ', len(test_preds))
            print('accu: ', test_accuracy, 'sen: ', sen, 'prec: ', prec, 'f1: ', f1)


            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            # if maxauc < auc_value:
            #     print('save model')
            #     torch.save(model.state_dict(), 'checkpoint/new_model_4.pkl')
            #     maxauc = auc_value
