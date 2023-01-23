# -*- coding: utf-8 -*-
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree

wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

clf1 = RandomForestClassifier(criterion = 'gini', n_estimators=20, random_state=42, n_jobs=(8))
clf1.fit(X_train,y_train)
clf2 = tree.DecisionTreeClassifier(criterion='entropy')
clf2.fit(X_train, y_train)

y_pred1=clf1.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
print(y_pred1)
y_pred2=clf2.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
print(y_pred2)

conf_mat1 = confusion_matrix(y_test, y_pred1)
print(conf_mat1)
print(classification_report(y_test, y_pred1))
conf_mat2 = confusion_matrix(y_test, y_pred2)
print(conf_mat2)
print(classification_report(y_test, y_pred2))

X_new = [[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]]
prediction1 = clf1.predict(X_new)
print(wine.target_names[prediction1])
prediction2 = clf2.predict(X_new)
print(wine.target_names[prediction2])
