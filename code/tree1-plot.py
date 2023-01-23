from sklearn.datasets import load_wine
from sklearn import tree

wine = load_wine()
print(wine.DESCR)

x = wine.data
y = wine.target

dtc = tree.DecisionTreeClassifier(criterion='entropy')
dtc = dtc.fit(x, y)
tree.plot_tree(dtc) 




