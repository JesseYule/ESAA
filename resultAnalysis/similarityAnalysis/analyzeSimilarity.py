import numpy as np
import pandas as pd
import os
import torch

# 计算两两circRNA向量之间的距离，用于分析相似度等等

circ = np.loadtxt("circRNA_embedding.csv", delimiter=',')
id = pd.read_excel("unique_circRNA.xlsx", header=0)
id = id["circRNA"]

result = pd.DataFrame(columns=['circRNA 1', 'circRNA 2', 'distance'])

for i in range(742):
    print(i)
    for j in range(i+1, 742):
        distance = np.linalg.norm(circ[i] - circ[j])
        result = result.append([{'circRNA 1': id[i], 'circRNA 2': id[j], 'distance': distance}])

print("finish calculation")

result.to_csv("circRNA_distance.csv")
