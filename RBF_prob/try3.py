import numpy as np
import pandas as pd
from save_model import LSTMQuantileRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import seaborn as sns

data = pd.read_excel(r'C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx')
x_data = data["电力（万千瓦时）"]
y_data = data["碳排放"]

#划分训练集、测试集

train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

#标准化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.fit_transform(y_test)
quantiles = np.arange(0.01,1,0.01)

predictions = []
huatu = []
for quantile in quantiles:
    model = LSTMQuantileRegressor(q=quantile,n_units=32)
    #model.fit(x_train,y_train,x_test,y_test)
    y_pred = model.predict(x_test)
    huatu.append(y_pred)
    y_pred = scaler_y.inverse_transform(y_pred)
    predictions.append(y_pred)
predictions = np.dstack(predictions)
predictions = predictions.reshape(len(x_test),len(quantiles))
huatu = np.dstack(huatu).T
huatu = huatu.reshape(len(x_test),len(quantiles))


density_list = []
for i in range(len(predictions)):
    gmm = GaussianMixture(n_components=3, max_iter=200)
    gmm.fit(predictions[i].reshape(-1, 1))
    density = np.exp(gmm.score_samples(predictions[i].reshape(-1, 1)))
    density_list.append(density)

density_list=np.vstack(density_list)

low_bond=[]
up_bond=[]
for i in range(len(predictions)):
    up = np.max(predictions[i])
    low = np.min(predictions[i])
    up_bond.append(up)
    low_bond.append(low)


# 画出真实值曲线
plt.plot(scaler_y.inverse_transform(y_test), label='True values')

# 绘制置信区间
plt.fill_between(np.arange(len(y_test)), low_bond, up_bond, color='gray', alpha=0.5)
plt.show()
'''
# 绘制热力图
plt.figure(figsize=(10,6))
sns.heatmap(huatu, cmap="YlGnBu",  xticklabels=False)
plt.ylabel("Quantile")
plt.xlabel("Test Samples")
plt.title("Density Heatmap")
plt.show()'''
