import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

# 首先要构建负样本

# data = pd.read_csv("circ_dis_association.csv")
# # data = data[['circRNA_index', 'disease_index']]
# #
# # result = pd.DataFrame(columns=["circRNA_index", "disease_index"])
# #
# # sum = 0
# #
# # while sum < 846:
# #
# #     temp1 = np.random.randint(741)
# #     temp2 = np.random.randint(71)
# #
# #     if len(data.loc[(data.circRNA_index == temp1) & (data.disease_index == temp2)]) == 0:
# #         result = result.append(
# #                     pd.DataFrame({'circRNA_index': [temp1],  'disease_index': [temp2]}),
# #                     ignore_index=True)
# #         sum += 1
# #
# # result.to_csv("negative_input.csv", index=False)

# 把正负样本的向量抽出来

circRNA_embedding_vector = np.loadtxt('circRNA_embedding.csv', delimiter=',')
circRNA_embedding_vector = np.mat(circRNA_embedding_vector)
disease_embedding_vector = np.loadtxt('disease_embedding.csv', delimiter=',')
disease_embedding_vector = np.mat(disease_embedding_vector)

# 先转正样本

positive_input = pd.read_csv("positive_input.csv")
positive_circRNA_index = positive_input['circRNA_index']
positive_disease_index = positive_input['disease_index']

positive_circRNA_vector = np.zeros(shape=(846, 300))  # 正样本有846条数据，每个向量长300
positive_disease_vector = np.zeros(shape=(846, 300))

for i in range(positive_circRNA_index.shape[0]):
    positive_circRNA_vector[i] = circRNA_embedding_vector[positive_circRNA_index[i]]
    positive_disease_vector[i] = disease_embedding_vector[positive_disease_index[i]]


# 再转负样本

negative_input = pd.read_csv("negative_input.csv")
negative_circRNA_index = negative_input['circRNA_index']
negative_disease_index = negative_input['disease_index']

negative_circRNA_vector = np.zeros(shape=(846, 300))  # 正样本有846条数据，每个向量长300
negative_disease_vector = np.zeros(shape=(846, 300))

for i in range(negative_circRNA_index.shape[0]):
    negative_circRNA_vector[i] = circRNA_embedding_vector[negative_circRNA_index[i]]
    negative_disease_vector[i] = disease_embedding_vector[negative_disease_index[i]]


np.savetxt("positive_circRNA_vector.csv", positive_circRNA_vector, delimiter=',')
np.savetxt("positive_disease_vector.csv", positive_disease_vector, delimiter=',')
np.savetxt("negative_circRNA_vector.csv", negative_circRNA_vector, delimiter=',')
np.savetxt("negative_disease_vector.csv", negative_disease_vector, delimiter=',')

