# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

model = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
#linkage: ['ward', 'complete', 'average']
model.fit(X)

labels = model.fit_predict(X)
#results visualization
plt.figure()
plt.scatter(X[:,0], X[:,1], c = labels)
plt.axis('equal')
plt.title('Prediction')
plt.show()


# Performs hierarchical/agglomerative clustering on X by using "Ward's method"
linkage_matrix = linkage(X, 'ward')
figure = plt.figure(figsize=(7.5, 5))
# Plots the dendrogram
dendrogram(linkage_matrix, truncate_mode = 'level', p=3, labels = labels)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.tight_layout()
plt.show()