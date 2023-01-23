# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import KFold

iris = load_iris()
X = iris.data
y = iris.target

clf = tree.DecisionTreeClassifier()

kf = KFold(n_splits=10)
kf.get_n_splits(X)
print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("TRAIN data:")
    print(X_train, y_train)
    print("TEST data:")
    print(X_test, y_test )
    clf = clf.fit(X_train, y_train)
print(clf.score(X, y))
    
