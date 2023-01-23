# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target
rng = np.random.RandomState(42)
n_total_samples = len(y)
indices = np.arange(n_total_samples)

random_unlabeled_points = rng.rand(n_total_samples) < 0.3
new_X, new_y = X[random_unlabeled_points], y[random_unlabeled_points]
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.3, random_state=0)

model = RandomForestClassifier(verbose = 0, max_depth=2, random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy (after Label Spreading): ",'{:.2%}'.format(acc))

