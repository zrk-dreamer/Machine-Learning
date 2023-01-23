# -*- coding: utf-8 -*-
from sklearn import cluster, datasets, metrics
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_X = iris.data

silhouette_avgs = []
ks = range(2, 11)

for k in ks:
    kmeans= cluster.KMeans(n_clusters = k)
    kmeans.fit(iris_X)
    cluster_labels = kmeans.labels_
    silhouette_avg = metrics.silhouette_score(iris_X, cluster_labels)
    silhouette_avgs.append(silhouette_avg)
    
print(silhouette_avgs)
plt.plot(ks, silhouette_avgs, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('silhouette_avgs')
plt.title('Elbow Method For Optimal k')
plt.show()
'''
plt.plot(ks, wcss, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()
'''
