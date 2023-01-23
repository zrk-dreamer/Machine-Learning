# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import cluster, metrics
import matplotlib.pyplot as plt

df=pd.read_csv(r"D:\Machine Learning\資料檔\zoo.csv", names=['name','hair', 'feathers','eggs','milk',
                                 'airborne', 'aquatic','predator','toothed','backbone',
                                 'breathes', 'venomous', 'fins', 'legs', 'tail', 
                                 'domestic','catsize','type'])
dfs = df 
X = df.drop('name', axis = 1)

kmeans_fit = cluster.KMeans(n_clusters = 7).fit(X)

cluster_labels = kmeans_fit.labels_
print("分群結果：", cluster_labels)
'''
dfs['cluster'] = cluster_labels
print(dfs)
dfs.to_csv('out1.csv', index = False)
'''
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

