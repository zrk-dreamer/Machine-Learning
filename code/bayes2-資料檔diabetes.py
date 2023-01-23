# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

df = pd.read_csv(r'D:\Machine Learning\資料檔\Naive-Bayes-Diabetes.csv')
X = df.drop('diabetes', axis = 1)
y = df['diabetes']
X_train,X_test,y_train,y_test=train_test_split(X, y, stratify=y, random_state=42)

bayes = GaussianNB(var_smoothing=1e-8)
bayes.fit(X_train, y_train)

y_pred = bayes.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(bayes.score(X_test, y_pred))