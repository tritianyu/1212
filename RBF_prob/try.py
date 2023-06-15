import numpy as np
import pandas as pd
from save_model import LSTMQuantileRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
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
quantiles = np.arange(0.01,1,0.1)

predictions = []
for quantile in quantiles:
    model = LSTMQuantileRegressor(q=quantile,n_units=32)
    #model.fit(x_train,y_train,x_test,y_test)
    y_pred = model.predict(x_test)
    y_pred = scaler_y.inverse_transform(y_pred)
    predictions.append(y_pred)
predictions = np.dstack(predictions)
predictions = predictions.reshape(len(x_test),len(quantiles))

idx = 55  # 假设要查看第个样本点对应的概率密度
point = predictions[idx, :]
n_quantiles = 1000  # 新的横坐标数量
quantiles_res = np.linspace(np.min(point), np.max(point), n_quantiles)
point.sort()
'''
#采用正态分布
loc = np.mean(point)
scale = np.std(point)
density = norm.pdf(point, loc=loc, scale=scale)
'''
'''
#采用混合高斯
gmm = GaussianMixture(n_components=3,max_iter=200,random_state=1)
gmm.fit(np.array(point).reshape(-1,1))
density = np.exp(gmm.score_samples(quantiles_res.reshape(-1,1)))


'''
#采用 kernel density estimation 方法生成概率密度函数
kde_model = KernelDensity(kernel='gaussian', bandwidth=500)
kde_model.fit(np.array(point).reshape(-1,1))
density = np.exp(kde_model.score_samples(quantiles_res.reshape(-1,1)))

print(density)
# 可视化展示概率密度函数、实际值和模型预测值
fig , ax = plt.subplots(figsize=(12, 6))
ax.plot(quantiles_res, density, label='probability density function')
y_test = scaler_y.inverse_transform(y_test)
ax.vlines(y_test[idx], 0, density.max()*1.2, label='true value', color='r')
plt.xlabel('Carbon Emissions')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
