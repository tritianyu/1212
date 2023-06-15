import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import sklearn.linear_model as LM
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.preprocessing import StandardScaler
N=100
#X,Y=make_regression(n_samples=N,n_features=1,random_state=123,noise=50,bias=0)
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电力、热力的生产和供应业月份数据.xlsx")
X=data["电力（万千瓦时）"].values.reshape(-1,1)
Y=data["碳排放"].values.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.85,random_state=123)
'''
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)

ss_y = StandardScaler()
Y_train = ss_y.fit_transform(Y_train)
Y_test = ss_y.transform(Y_test)
'''
plt.scatter(X_train,Y_train,s=20)
plt.scatter(X_test,Y_test,s=20,marker='*')
plt.title("样本观测点的SVR和线性回归")
plt.xlabel("X")
plt.ylabel("Y")
modelLM=LM.LinearRegression()
modelLM.fit(X_train,Y_train)
X[:,0].sort()
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(12,9))
for C,E,H,L in [(10000,0.1,0,0),(10000,1,0,1),(100000,0.1,1,0),(10000,0.01,1,1)]:
    modelSVR=svm.SVR(C=C,epsilon=E)
    modelSVR.fit(X_train,Y_train)
    axes[H,L].scatter(X_train,Y_train,s=20)
    axes[H, L].scatter(X_test,Y_test, s=20,marker='*')
    #axes[H, L].scatter(X[modelSVR.support_],Y[modelSVR.support_],marker='o',c='b',s=120,alpha=0.2)
    axes[H, L].plot(X,modelSVR.predict(X),linestyle='-',label='SVR')
    axes[H, L].plot(X,modelLM.predict(X),linestyle='--',label='linear',linewidth=1)
    axes[H, L].legend()
    ytrain=modelSVR.predict(X_train)
    ytest=modelSVR.predict(X_test)
    axes[H, L].set_title("SVR(C=%d,epsilon=%.2f,训练误差=%.2f,测试MSE=%.2f)"%(C,E,mean_squared_error(Y_train,ytrain),mean_squared_error(Y_test,ytest)))
    axes[H, L].set_xlabel("X")
    axes[H, L].set_ylabel("Y")
    axes[H, L].grid(True,linestyle='-.')
plt.show()