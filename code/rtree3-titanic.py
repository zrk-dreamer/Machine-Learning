# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

titanic = pd.read_csv('titanic_train.csv')
label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic['Sex'])
titanic['Sex'] = encoded_Sex

age_median = np.nanmedian(titanic['Age'])
new_Age = np.where(titanic['Age'].isnull(), age_median, titanic['Age'])
titanic['Age'] = new_Age

X = titanic[['Pclass', 'Sex', 'Age']]
y = titanic['Survived']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)


clf=RandomForestClassifier(criterion = 'gini', n_estimators=20, random_state=42, n_jobs=(8))
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(y_pred)
print(clf.estimators_)
print(clf.predict_proba(X_test))
