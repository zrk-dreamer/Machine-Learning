# -*- coding: utf-8 -*-
#載入套件&讀入資料
from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import graphviz
import pickle

#觀察資料內容
wine = load_wine()
print(wine.data.shape)
print(wine.data[0])
print(wine.target.shape)
print(wine.target)
print(wine.target_names)
print(wine.DESCR)

#建立模型並訓練
X = wine.data
y = wine.target
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

#列印錯差矩陣及其性能指標
y_pred = cross_val_predict(clf, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))

#列印決策樹及其規則
dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("wine") 
'''
#列印決策樹及其規則(美化版)
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=wine.feature_names,  
                     class_names=wine.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("winevis") 
'''
#儲存model
pkl_filename = "wine_model.pkl" 
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)

with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

#用儲存的model預測新案例的分類結果
newX = [[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]]
print(pickle_model.predict(newX))





