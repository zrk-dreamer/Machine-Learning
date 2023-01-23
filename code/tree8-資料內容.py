# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)
print(iris.data[0])
print(iris.target.shape)
print(iris.target)
print(iris.target_names)
print(iris.DESCR)
