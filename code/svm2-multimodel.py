# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test=train_test_split(X, y, stratify=y, random_state=42)

models = (svm.SVC(kernel='linear'),
                    svm.SVC(kernel='rbf', gamma=0.7),
                    svm.SVC(kernel='poly', degree=3, gamma='auto'))


for clf in models:
    models = clf.fit(X_train, y_train)
    y_pred = models.predict(X_test)

    #查看各项得分
    print("y_pred",y_pred)
    print("y_test",y_test)
    print("score on train set", models.score(X_train, y_train))
    print("score on test set", models.score(X_test, y_test))
    print("accuracy score", accuracy_score(y_test, y_pred))
    # 可加上 confusion matrix & report    
    # What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
    X_new = [[3, 5, 4, 2]]
    prediction = models.predict(X_new)
    print(iris.target_names[prediction])
