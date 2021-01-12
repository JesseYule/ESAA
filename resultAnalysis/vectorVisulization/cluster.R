# Title     : TODO
# Objective : TODO
# Created by: junjieyu
# Created on: 2020-08-17

library(factoextra)
pca <- read_csv("pca.csv")
pca<-na.omit(pca)
km_result5 <- kmeans(pca, 5)
fviz_cluster(km_result5, data = pca,
              ellipse.type = "euclid",
              star.plot = TRUE,
              repel = TRUE,
              ggtheme = theme_minimal())
cluster5 <- cbind(pca, cluster = km_result5$cluster)
write.csv(cluster5,file="cluster15.csv",quote=F,row.names = F)