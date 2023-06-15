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
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\分月份数据.xlsx")
X=data["电力（万千瓦时）"].values.reshape(-1,1)
Y=data["碳排放"].values.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.85,random_state=123)
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)
ss_y = StandardScaler()
Y_train = ss_y.fit_transform(Y_train)
Y_test = ss_y.transform(Y_test)
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(12,9))
for C,E,H,L in [(1,0.1,0,0),(1,100,0,1),(100,0.1,1,0),(10000,0.01,1,1)]:
    modelSVR=svm.SVR(C=C,epsilon=E)
    modelSVR.fit(X_train,Y_train)
    axes[H,L].scatter(X_train,Y_train,s=20)
    plt.show()