#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns

iris = datasets.load_iris()
print(iris.DESCR)
iris = sns.load_dataset("iris")   #not from load_iris()
print(iris.head())

sns.countplot('species',data=iris)
plt.show()

sns.jointplot("sepal_length", "sepal_width", data=iris)
sns.jointplot("petal_length", "petal_width", data=iris)
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='sepal_width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='petal_width',data=iris)
plt.show()

sns.set()
sns.pairplot(iris, hue='species', height=3)
plt.show()

sns.set_style("whitegrid")
sns.FacetGrid(iris, hue ="species", height = 6).map(plt.scatter,
                  'sepal_length',  'petal_length').add_legend()

sns.set_style("whitegrid")
sns.FacetGrid(iris, hue ="species", height = 6).map(plt.scatter,
                  'sepal_width',  'petal_width').add_legend()
plt.show()