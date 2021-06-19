# ESAA: Embedding and Semantic Association Analysis between CircRNA and Disease by Deep Learning

### Dataset

CircR2Disease: http://bioinfo.snnu.edu.cn/

DisGeNET: https://www.disgenet.org/

### Supporting information

S1 Table. The processed circR2Disease dataset contains 580 circRNA and 71 diseases.

S2 Table. The processed DisGeNET dataset contains 71 diseases and 12092 genes.



### Model Training

1. ESAA/disease_embedding

This folder is about the calculation of disease embedding vector. ESAA/disease_embedding/preprocess/processData.py is used to preprocess the data to obtain the adjacency matrix.



In ESAA/disease_embedding/model, we can run train.py to train the ERN. After overfitting the model, we can run saveEmbeddingVector.py to obtain the embedding vector of disease.



2. ESAA/circrna_embedding

The procedure is similar to the embedding of disease.



3. ESAA/circ_dis_analysis

Running ESAA/circ_dis_analysis/preprocess/constructInput.py to obtain the input of siamese network.

I have put the processed input files in ESAA/circ_dis_analysis/model, so we can run ESAA/circ_dis_analysis/model/train.py directly to train our model.
