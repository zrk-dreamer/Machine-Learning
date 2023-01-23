import numpy as np 
from sklearn.linear_model import LinearRegression 

X = np.array([ [10, 80], [8, 0], [8, 200], [5, 200], [7, 300], [8, 230], [7, 40], [9, 0], [6, 330], [9, 180] ]) 
y = np.array([469, 366, 371, 208, 246, 297, 363, 436, 198, 364]) 
reg= LinearRegression() 
reg.fit(X, y) 		
print(u'係數', reg.coef_)
print (u'截距', reg.intercept_)
print (u'評分函式', reg.score(X, y))
print('The residual sum of squares: {:.2f}'.format(np.mean((reg.predict(X)-y)** 2)))
predicted = np.array([ [10, 110] ]) 
predicted_sales = reg.predict(predicted)
print("%d" % predicted_sales)


