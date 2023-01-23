# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test=train_test_split(X, y, stratify=y, random_state=42)

svm = svm.SVC(kernel = 'linear', random_state = 42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
conf = confusion_matrix(y_test, y_pred)
print(conf)
print(classification_report(y_test, y_pred))
print(svm.score(X_test, y_pred))
print(accuracy_score(y_test, y_pred))