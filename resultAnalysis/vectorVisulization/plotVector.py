# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
import pandas as pd

vector = np.loadtxt("positive_circRNA_vector.csv", delimiter=',')

circRNAname = pd.read_csv('unique_circRNA.csv', header=0)
circRNAname = circRNAname['circRNA'].values.tolist()

pca = PCA(n_components=2)
result = pca.fit_transform(vector)
# result = result * 10
result = np.array(result)
# np.savetxt("pca.csv", result, delimiter=',')

# 可视化展示
pyplot.figure(figsize=(30, 30))
pyplot.xlim(-1, 4)
pyplot.ylim(-2, 3)

pyplot.scatter(result[:, 0], result[:, 1])

words = circRNAname
# print(words)

for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()
