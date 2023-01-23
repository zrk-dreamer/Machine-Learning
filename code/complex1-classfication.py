# -*- coding: utf-8 -*-
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree, svm, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

models = (tree.DecisionTreeClassifier(criterion='entropy'),
          RandomForestClassifier(criterion = 'entropy', n_estimators=20, random_state=42),
          svm.SVC(kernel='poly', degree=3, gamma='auto'),
          GaussianNB(var_smoothing=1e-8),
          neighbors.KNeighborsClassifier(n_neighbors=5,weights='uniform', metric='minkowski')
          )

modelL1 = list()
for model in models:
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print('R2:',model.score(X_test, y_test))
    print('精確度:',accuracy_score(y_test, y_pred))
    print('混淆矩陣:\n',confusion_matrix(y_test, y_pred))
    print('綜合報告:\n',classification_report(y_test, y_pred))
    modelL1.append(accuracy_score(y_test, y_pred))
    
    
    
    
iris = load_iris()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

models = (tree.DecisionTreeClassifier(criterion='entropy'),
          RandomForestClassifier(criterion = 'entropy', n_estimators=20, random_state=42),
          svm.SVC(kernel='poly', degree=3, gamma='auto'),
          GaussianNB(var_smoothing=1e-8),
          neighbors.KNeighborsClassifier(n_neighbors=5,weights='uniform', metric='minkowski')
          )

modelL2 = list()
for model in models:
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print('R2:',model.score(X_test, y_test))
    print('精確度:',accuracy_score(y_test, y_pred))
    print('混淆矩陣:\n',confusion_matrix(y_test, y_pred))
    print('綜合報告:\n',classification_report(y_test, y_pred))
    modelL2.append(accuracy_score(y_test, y_pred))
 
print('在wine資料中最出色的模型是:')
best1 = modelL1.index(max(modelL1))
if best1 == 0:
    print('tree!')
elif best1 == 1:
    print('random forest!')
elif best1 == 2:
    print('svm!')
elif best1 == 3:
    print('bayes!')
else:
    print('KNN!')
print('精確度:', max(modelL1))
 
print('在iris資料中最出色的模型是:')
best2 = modelL2.index(max(modelL2))
if best2 == 0:
    print('tree!')
elif best2 == 1:
    print('random forest!')
elif best2 == 2:
    print('svm!')
elif best2 == 3:
    print('bayes!')
else:
    print('KNN!')
print('精確度:', max(modelL2))
    