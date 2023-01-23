# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\Machine Learning\資料檔\Mall_Customers.csv')
df = df.drop(['CustomerID'], axis = 1)
X = df.iloc[:, [2, 3]].values

print(X.shape)
print(X)

eps = 3
min_samples = 4

dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

cluster_labels = dbscan.labels_
print("分群結果：", cluster_labels)

# 印出績效
silhouette_avg = metrics.silhouette_score(X, cluster_labels)
print(silhouette_avg)

x0 = X[cluster_labels == 0]
x1 = X[cluster_labels == 1]
x2 = X[cluster_labels == 2]
x3 = X[cluster_labels == 3]
x4 = X[cluster_labels == 4]
x5 = X[cluster_labels == 5]
x6 = X[cluster_labels == 6]
x7 = X[cluster_labels == 7]
x8 = X[cluster_labels == 8]

plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c="orange", marker='.', label='label3')
plt.scatter(x4[:, 0], x4[:, 1], c="yellow", marker='^', label='label4')
plt.scatter(x5[:, 0], x5[:, 1], c="purple", marker='>', label='label5')
plt.scatter(x6[:, 0], x6[:, 1], c="black", marker='<', label='label6')
plt.scatter(x7[:, 0], x7[:, 1], c="gray", marker=',', label='label7')
plt.scatter(x8[:, 0], x8[:, 1], c="pink", marker='|', label='label8')  

plt.xlabel('Age')  
plt.ylabel('Annual Income')  
plt.legend(loc=4)  
plt.show()