#-*- coding: utf-8 -*-
from sklearn.datasets import load_wine
from sklearn import tree

wine = load_wine()
print(wine.DESCR)

x = wine.data
y = wine.target

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x, y)
tree.plot_tree(clf) 

