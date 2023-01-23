# -*- coding: utf-8 -*-
# import the Classifier
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

digits = datasets.load_digits()

X = digits.data
y = digits.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


clf = tree.DecisionTreeClassifier(criterion='entropy')
#clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

#查看各项得分
print("y_pred",y_pred)
print("y_test",y_test)
print("score on train set", clf.score(X_train, y_train))
print("score on test set", clf.score(X_test, y_test))
print("accuracy score", accuracy_score(y_test, y_pred))

#列印錯差矩陣及其性能指標
y_pred = cross_val_predict(clf, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))