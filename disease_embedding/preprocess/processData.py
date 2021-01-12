import pandas as pd
import numpy as np


# data = pd.read_csv('gene_disease.tsv', sep='\t')
# data = data[['diseaseName', 'geneId']]
#
# data = data.drop_duplicates()
# data = data.reset_index(drop=True)  # 删除重复数据之后一定要重置索引，不然后面计算会出错
#
# data['diseaseName'] = data['diseaseName'].str.lower()
#
# unique_disease = pd.read_excel('unique_disease(circR2Disease).xlsx')
# unique_disease = unique_disease['disease'].str.lower()
#
# # print(unique_disease[0])
#
# tmp = data[data['diseaseName'].str.contains(unique_disease[0])]
#
# result = pd.DataFrame(columns=["diseaseName", "geneId", "diseaseName2"])
#
# # circR2Disease一共有100种disease，找出每种disease在disgenet中对应的所有行，也就是每种disease对应的
# # gene，因为不是绝对相等，所以只要disgenet的disease包含了circR2Disease的disease的一部分都认为两者相同
#
# for i in range(len(unique_disease)):
#     print(i)
#     tmp = data[data['diseaseName'].str.contains(unique_disease[i])]
#     if len(tmp) > 0:
#         tmp['diseaseName2'] = unique_disease[i]
#         result = result.append(tmp, ignore_index=True)
#
# print(result)
# result.to_csv("gene_disease(circR2Disease).csv", index=None)

# 后续还要后处理，一方面是去除重复值，一方面是筛选出gene_disease和circR2Disease共有的disease

# 然后通过excel为disease和gene加上index
# gene_disease.csv就是gene_disease(circR2Disease).csv通过excel处理过的文件


data = pd.read_csv('gene_disease.csv')
disease = data['disease']
gene = data['gene']

disease = disease.drop_duplicates()
gene = gene.drop_duplicates()
print(len(disease))
print(len(gene))

# 有71种病，12092种gene

disease_index = data['disease index']
gene_index = data['gene index']

matrix = np.zeros([len(disease), len(gene)])

for i in range(len(data)):
    matrix[disease_index[i], gene_index[i]] = 1

print(matrix.shape)

np.savetxt("adjacency_matrix.csv", matrix, delimiter=',', fmt='%d')