# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]


reg = LinearRegression() 
reg.fit(X, y) 
print(u'係數', reg.coef_)
print (u'截距', reg.intercept_)
print (u'評分函式', reg.score(X, y))

X2 = [[1], [10], [14], [25]]
y2 = reg.predict(X2)
print("prediccted new: ")
print(y2)
#繪製線性迴歸圖形
plt.figure()
plt.title(u'Pizza Price with diameter.')   #標題
plt.xlabel(u'diameter')              #x軸座標
plt.ylabel(u'price')                  #y軸座標
plt.axis([0, 25, 0, 25])             #區間
plt.grid(True)                       #顯示網格
plt.plot(X, y, 'bo')                 #藍色點
plt.plot(X2, y2, color = 'r')              #繪製預測資料集直線

print('The residual sum of squares: {:.2f}'.format(np.mean((reg.predict(X)-y)** 2)))