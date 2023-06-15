import numpy as np
import warnings
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import zero_one_loss,r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
#N=1000
#X,Y=make_regression(n_samples=N,n_features=10,random_state=123)
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
X=data["电力（万千瓦时）"].values.reshape(-1,1)
Y=data["原煤"].values.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,random_state=123)
B=300
dt_stump=tree.DecisionTreeRegressor(max_depth=1,min_samples_leaf=1)
TrainErrAdaB=np.zeros((B,))
TestErrAdaB=np.zeros((B,))
adaBoost=ensemble.AdaBoostRegressor(base_estimator=dt_stump,n_estimators=B,loss='square',random_state=123)
adaBoost.fit(X_train,Y_train)
for b,Y_pred in enumerate(adaBoost.staged_predict(X_train)):
    TrainErrAdaB[b]=mean_squared_error(Y_train,Y_pred)
for b,Y_pred in enumerate(adaBoost.staged_predict(X_test)):
    TestErrAdaB[b]=mean_squared_error(Y_test,Y_pred)

GBRT=ensemble.GradientBoostingRegressor(loss='ls',n_estimators=B,max_depth=1,min_samples_leaf=1,random_state=123)
GBRT.fit(X_train,Y_train)
TrainErrGBRT=np.zeros((B,))
TestErrGBRT=np.zeros((B,))
for b,Y_pred in enumerate(GBRT.staged_predict(X_train)):
    TrainErrGBRT[b]=mean_squared_error(Y_train,Y_pred)
for b,Y_pred in enumerate(GBRT.staged_predict(X_test)):
    TestErrGBRT[b]=mean_squared_error(Y_test,Y_pred)

GBRT0=ensemble.GradientBoostingRegressor(loss='ls',n_estimators=B,max_depth=3,min_samples_leaf=1,random_state=123)
GBRT0.fit(X_train,Y_train)
TrainErrGBRT0=np.zeros((B,))
TestErrGBRT0=np.zeros((B,))
for b,Y_pred in enumerate(GBRT0.staged_predict(X_train)):
    TrainErrGBRT0[b]=mean_squared_error(Y_train,Y_pred)
for b,Y_pred in enumerate(GBRT0.staged_predict(X_test)):
    TestErrGBRT0[b]=mean_squared_error(Y_test,Y_pred)

plt.plot(np.arange(B),TrainErrAdaB,linestyle='--',label="AdaBoost回归树（训练）",linewidth=0.8)
plt.plot(np.arange(B),TestErrAdaB,linestyle='--',label="AdaBoost回归树（测试）",linewidth=2)
plt.plot(np.arange(B),TrainErrGBRT,linestyle='-',label="梯度提升回归树（训练）",linewidth=0.8)
plt.plot(np.arange(B),TestErrGBRT,linestyle='-',label="梯度提升回归树（测试）",linewidth=2)
plt.plot(np.arange(B),TrainErrGBRT0,linestyle='-',label="复杂梯度提升回归树（训练）",linewidth=0.8)
plt.plot(np.arange(B),TestErrGBRT0,linestyle='-',label="复杂梯度提升回归树（测试）",linewidth=2)
plt.title=("梯度提升回归树和AdaBoost回归树")
plt.xlabel("弱模型个数")
plt.ylabel("误差")
plt.legend()
plt.show()