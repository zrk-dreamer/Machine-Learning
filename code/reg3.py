import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn import datasets 

diabetes = datasets.load_diabetes() #載入資料
print(diabetes.DESCR)
X = diabetes.data
Y = diabetes.target

reg= LinearRegression()

reg.fit(X, Y) 		
print(u'係數', reg.coef_)
print (u'截距', reg.intercept_)
print (u'評分函式', reg.score(X, Y))
print('The residual sum of squares: {:.2f}'.format(np.mean((reg.predict(X)-Y)** 2)))