# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

clf=RandomForestClassifier(criterion = 'gini', n_estimators=2000000, random_state=42, n_jobs=(8))
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(y_pred)
print(clf.estimators_)
print(clf.predict_proba(X_test))






