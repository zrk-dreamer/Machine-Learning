# -*- coding: utf-8 -*-
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = datasets.load_iris()
#print(iris['DESCR'])
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

for i in range(3, 12, 2):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i,weights='uniform', metric='minkowski')
    knn.fit(X, y)
    y_pred = knn.predict(X_test)
    #print("y_pred",y_pred)
    #print("y_test",y_test)
    #print("score:", knn.score(X_test, y_test))
    print("accuracy score:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    
    X_new = [[3, 5, 4, 2]]
    prediction = knn.predict(X_new)
    print(iris.target_names[prediction])
