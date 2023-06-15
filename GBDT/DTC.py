import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pandas as pd
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
X=data["电力（万千瓦时）"].values.reshape(-1,1)
Y=data["碳排放"].values.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,random_state=123)
B=300

GBRT=ensemble.GradientBoostingRegressor(loss='ls',n_estimators=B,max_depth=1,min_samples_leaf=1,random_state=123)
GBRT.fit(X_train,Y_train)
Y_pred1=GBRT.predict(X_test)
err1=mean_absolute_percentage_error(Y_pred1,Y_test)
print(err1)
GBRT0=ensemble.GradientBoostingRegressor(loss='ls',n_estimators=B,max_depth=2,min_samples_leaf=2,random_state=123)
GBRT0.fit(X_train,Y_train)
Y_pred0=GBRT.predict(X_test)
err2=mean_absolute_percentage_error(Y_pred0,Y_test)
print(err2)
