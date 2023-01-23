# -*- coding: utf-8 -*-
from sklearn import datasets, metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

eps = 0.4
min_samples = 4

dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

cluster_labels = dbscan.labels_
print("分群結果：", cluster_labels)
# 印出品種看看
print("真實品種：", y)

# 印出績效
silhouette_avg = metrics.silhouette_score(X, cluster_labels)
print(silhouette_avg)

x0 = X[cluster_labels == 0]
x1 = X[cluster_labels == 1]
x2 = X[cluster_labels == 2]
plt.scatter(x0[:, 2], x0[:, 3], c="red", marker='o', label='label0')  
plt.scatter(x1[:, 2], x1[:, 3], c="green", marker='*', label='label1')  
plt.scatter(x2[:, 2], x2[:, 3], c="blue", marker='+', label='label2')  
plt.xlabel('petal length')  
plt.ylabel('petal width')  
plt.legend(loc=2)  
plt.show() 
