# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

#Load Data
df = pd.read_csv("iris_unlables.csv")
df = df.sort_values(by = ["species"], ascending=False, ignore_index=True)
X = df.drop(["speciesTrue","species"], axis = 1)
y = df[["speciesTrue","species"]]
y_true = y["speciesTrue"]
y_unlabel_train = y["species"].copy()

n_total_samples = len(y_unlabel_train)
indices = np.arange(n_total_samples)
unlabeled_set = indices[np.where(y_unlabel_train == -1)] 
#fit to Label Spreading
label_spread_model = LabelSpreading(gamma=.25, max_iter=20)
label_spread_model.fit(X, y_unlabel_train)

# Predict the Labels for Unlabeled Samples
spread_labels = label_spread_model.transduction_[unlabeled_set]
true_labels = y_true[unlabeled_set]
print(true_labels)
print(spread_labels)
cm = confusion_matrix(true_labels, spread_labels, labels=label_spread_model.classes_)
print(classification_report(true_labels, spread_labels))
print("Confusion matrix")
print(cm)

y_spread_train = y_unlabel_train
y_spread_train[unlabeled_set] = spread_labels
#------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_unlabel_train, test_size=0.3, random_state=0)

model = RandomForestClassifier(verbose = 0, max_depth=2, random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy (after Label Spreading): ",'{:.2%}'.format(acc))
