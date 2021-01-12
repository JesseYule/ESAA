import pandas as pd
import numpy as np

cluster = pd.read_csv('cluster15.csv', header=0)
circdisease = pd.read_csv('circid-disease.csv', header=0)

# print(cluster)
# print(circdisease)

result = pd.DataFrame(columns=['circRNA', 'disease', 'cluster'])


for i in range(len(circdisease)):
    circrna = circdisease['circRNA'][i]
    disease = circdisease['disease'][i]
    temp = cluster.loc[cluster.circRNA == circrna]
    clustervalue = temp['cluster'].values
    clustervalue = int(clustervalue)
    result = result.append(
        pd.DataFrame({'circRNA': [circrna], 'disease': [disease], 'cluster': [clustervalue]}),
        ignore_index=True)

result.to_csv('circDiseaseCluster15.csv', index=False)