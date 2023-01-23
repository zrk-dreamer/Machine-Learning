# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

wine = load_wine()
X = wine.data
y = wine.target
X_train,X_test,y_train,y_test=train_test_split(X, y, stratify=y, random_state=42)

bayes = GaussianNB(var_smoothing=1e-8)
bayes.fit(X_train, y_train)

y_pred = bayes.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print(classification_report(y_test, y_pred))

X_new = [[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]]
prediction = bayes.predict(X_new)
print(wine.target_names[prediction])

