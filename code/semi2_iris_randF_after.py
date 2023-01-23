# -*- coding: utf-8 -*-
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#Load Data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Random Number Generator
rng = np.random.RandomState(42)
n_total_samples = len(iris.target)
label_spread_model = LabelSpreading()

#Define How Many Samples Should be Unlabeled
random_unlabeled_points = rng.rand(len(y)) <= 0.3 #Almost 50% samples are unlabeled
n_labeled_points = 40
indices = np.arange(n_total_samples)
unlabeled_set = indices[n_labeled_points:]
#Seperate list for Unlabeled Samples
Unlabeled = np.copy(iris.target)
Unlabeled[random_unlabeled_points] = -1

#fit to Label Spreading
label_spread_model.fit(iris.data, Unlabeled)

# Predict the Labels for Unlabeled Samples
pred_lb = label_spread_model.transduction_[unlabeled_set]
new_y = y.copy()
print(new_y)
new_y[n_labeled_points:] = pred_lb
print(new_y)
#Accuracy of Prediction
print("Accuracy of Label Spreading: ",'{:.2%}'.format(label_spread_model.score(X, new_y)))
#------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, new_y, test_size=0.3, random_state=0)

model = RandomForestClassifier(verbose = 0, max_depth=2, random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy (after Label Spreading): ",'{:.2%}'.format(acc))
