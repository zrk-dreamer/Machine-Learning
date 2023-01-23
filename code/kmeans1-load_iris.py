# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, metrics

iris = datasets.load_iris()
X = iris.data

# KMeans 演算法
kmeans_fit = cluster.KMeans(n_clusters = 150).fit(X)

# 印出分群結果
cluster_labels = kmeans_fit.labels_
print("分群結果：", cluster_labels)

# 印出品種看看
y = iris.target
print("真實品種：", y)

# 印出績效
silhouette_avg = metrics.silhouette_score(X, cluster_labels)
print(silhouette_avg)
'''
for i in cluster_labels:
    print(i)
'''    
plt.subplot(2,2,1)
plt.scatter(cluster_labels, X[:,0],c = 'red')
plt.subplot(2,2,2)
plt.scatter(cluster_labels, X[:,1],c = 'yellow')
plt.subplot(2,2,3)
plt.scatter(cluster_labels, X[:,2],c = 'blue')
plt.subplot(2,2,4)
plt.scatter(cluster_labels, X[:,3],c = 'purple')
plt.show()
plt.scatter(X[:,0],X[:,1],c=cluster_labels)
plt.show()
plt.scatter(X[:,2],X[:,3],c=cluster_labels)
plt.show()

