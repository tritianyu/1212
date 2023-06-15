import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电气机械及器材制造业.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=33)
x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)
y_train=y_train.values.reshape(-1,1)
y_test=y_test.values.reshape(-1,1)
#创建模型对象
regr=LinearRegression()
regr.fit(x_train,y_train)
print(regr.coef_,regr.intercept_)
lr_pred=regr.predict(x_test)
ridge=Ridge()
ridge.fit(x_train,y_train)
ridge_pred=ridge.predict(x_test)
plt.scatter(x_test,y_test,c='b')
plt.scatter(x_test,lr_pred,c='g',s=100)
plt.scatter(x_test,ridge_pred,c='r')
plt.show()

