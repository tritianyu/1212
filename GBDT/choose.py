import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score,train_test_split
import pandas as pd
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
X_data=data["电力（万千瓦时）"].values.reshape(-1,1)
Y_data=data["原煤"].values.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.2,random_state=33)
trainErr=[]
testErr=[]
CVErr=[]
K=np.arange(2,15)
for k in K:
    modelDTC=tree.DecisionTreeRegressor(max_depth=k,random_state=123)
    modelDTC.fit(X_train,Y_train)
    trainErr.append(1-modelDTC.score(X_train,Y_train))
    testErr.append(1 - modelDTC.score(X_test, Y_test))
    Err=1-cross_val_score(modelDTC,X_data,Y_data,cv=5,scoring='r2')
    CVErr.append(Err.mean())

fig=plt.figure(figsize=(15,6))
ax1=fig.add_subplot(121)
ax1.grid(True,linestyle='-.')
ax1.plot(K,trainErr,label="训练误差",marker='o',linestyle='-')
ax1.plot(K,testErr,label="旁制法测试误差",marker='o',linestyle='-')
ax1.plot(K,CVErr,label="5-折交叉验证测试误差",marker='o',linestyle='--')
ax1.set_xlabel("树深度")
ax1.set_ylabel("误差（1-R方）")
ax1.set_title('树深度和误差')
ax1.legend()
plt.show()