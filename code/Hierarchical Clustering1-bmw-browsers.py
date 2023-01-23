# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv(r'D:\Machine Learning\資料檔\bmw-browsers.csv')
dfs = df

X = df.drop('CustomerID', axis = 1)
clustering = AgglomerativeClustering(n_clusters = 5, linkage = 'ward') 
clustering.fit(X)

cluster_labels = clustering.labels_
print(clustering.labels_)

silhouette_avg = metrics.silhouette_score(X, clustering.labels_)
print(silhouette_avg)

dfs['cluster'] = cluster_labels
print(dfs)
dfs.to_csv('out2.csv', index = False)

labels = clustering.fit_predict(X)

'''
plt.figure()
plt.scatter(X[:,0], X[:,1], c = labels)
plt.axis('equal')
plt.title('Prediction')
plt.show()
'''

linkage_matrix = linkage(X, 'ward')
figure = plt.figure(figsize=(7.5, 5))

dendrogram(linkage_matrix, truncate_mode = 'level', p=3, labels = labels)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show()



