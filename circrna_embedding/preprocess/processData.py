import pandas as pd
import numpy as np

# 把alias分成多行，每行表示一个circRNA

data = pd.read_csv('association2.csv', header=0)
circRNA = data['circRNA']
region = data['Region']

result = pd.DataFrame(columns=["circRNA", "Region"])

for i in range(len(circRNA)):
    if "/" in circRNA[i]:
        tmp = circRNA[i].split("/")
        for j in range(len(tmp)):
            result = result.append(
                pd.DataFrame({'circRNA': [tmp[j]], 'Region': [region[i]]}),
                ignore_index=True)
    else:
        result = result.append(
            pd.DataFrame({'circRNA': [circRNA[i]],  'Region': [region[i]]}),
            ignore_index=True)

result = result.drop_duplicates()
circRNA = result['circRNA']
region = result['Region']
result.to_csv("circRNA_info.csv", index=None, encoding='utf_8_sig')

# 这里用excel处理一下，加上index

# 构建adjacency matrix

# data = pd.read_csv('edited_association.csv', header=0)
#
#
# circRNA = data['circRNA']
# disease = data['disease']
#
# circRNA = circRNA.drop_duplicates()
# disease = disease.drop_duplicates()
# circRNA = circRNA.reset_index(drop=True)
# disease = disease.reset_index(drop=True)
#
# # 有742种circRNA，71种病
#
# circRNA_index = data['circRNA index']
# disease_index = data['disease index']
#
# matrix = np.zeros([len(circRNA), len(disease)])
#
# for i in range(len(data)):
#     matrix[circRNA_index[i], disease_index[i]] = 1
#
# print(matrix.shape)
#
# np.savetxt("adjacency_matrix.csv", matrix, delimiter=',', fmt='%d')
