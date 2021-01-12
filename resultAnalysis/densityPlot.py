import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# prob1 = pd.read_csv("hcc.csv", header=0, usecols=[1])
# prob2 = pd.read_csv("breast_cancer.csv", header=0, usecols=[1])
#
# prob1 = prob1.values
# prob1 = np.squeeze(prob1)
#
# prob2 = prob2.values
# prob2 = np.squeeze(prob2)
#
# plt.plot(prob2, label='breast cancer')
# plt.plot(prob1, linestyle=':', label='HCC')
#
# plt.ylabel('probability')
# plt.xlabel('no. of circRNA')
#
# plt.legend()
# plt.show()

# 对预测数据进行分组，观察每组预测准确的数量，从而分析整体的分布

data = pd.read_csv("hcc.csv", header=0, usecols=[4])
data = data.values
data = np.squeeze(data)

print(data.shape)
density = []
x_axis = []
for i in range(len(data) - 40):
    if i % 40 == 0:
        density.append(sum(data[i:i+40]))
        x_axis.append(i)
    if i == 701:
        density.append(sum(data[701:741]))
        x_axis.append(i)

print(density)

plt.hist(density, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)

plt.show()