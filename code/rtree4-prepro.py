# -*- coding: utf-8 -*-
#載入套件&讀入資料
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import graphviz

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
print(type(y))

#資料前處理
print(y.shape)
target=y.reshape(-1,1)
print(target.shape)
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
yb = est.fit_transform(target)

#分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, yb, test_size=0.7)

clf=RandomForestClassifier(criterion = 'gini', n_estimators=20, random_state=42, n_jobs=(8))
clf = clf.fit(X_train, y_train)

#列印錯差矩陣及其性能指標
y_pred=clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
print(classification_report(y_test, y_pred))


'''
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("wine") 
'''