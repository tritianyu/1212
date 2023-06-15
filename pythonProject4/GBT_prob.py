import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from GBT import GBTQuantileRegressor
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings(action='ignore')
# 加载电力数据和碳排放数据
def get_data():
    data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电气机械及器材制造业.xlsx")
    x_data = data["电力（万千瓦时）"]
    y_data = data["碳排放"]
    return data, x_data, y_data


data, x_data, y_data = get_data()
#划分训练集、测试集
def split(data, x_data, y_data):
    train_size = int(len(data) * 0.8)
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split(data, x_data, y_data)

#标准化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_train = y_train.ravel()
y_test = scaler_y.fit_transform(y_test)

def get_prediction():
    quantiles = np.arange(0.01, 1, 0.01)
    predictions = []
    for quantile in quantiles:
        model = GBTQuantileRegressor(q=quantile)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = y_pred.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred)
        predictions.append(y_pred)
    predictions = np.dstack(predictions)
    predictions = predictions.reshape(len(x_test), len(quantiles))
    return predictions


predictions = get_prediction()

idx = 49  # 假设要查看第个样本点对应的概率密度
point = predictions[idx, :]
n_quantiles = 1000  # 新的横坐标数量
quantiles_res = np.linspace(np.min(point), np.max(point), n_quantiles)
point.sort()

#采用正态分布
loc = np.mean(point)
scale = np.std(point)
density0 = norm.pdf(quantiles_res, loc=loc, scale=scale)

#采用混合高斯
gmm = GaussianMixture(n_components=3, max_iter=200, random_state=1)
gmm.fit(np.array(point).reshape(-1, 1))
density1 = np.exp(gmm.score_samples(quantiles_res.reshape(-1, 1)))


#采用 kernel density estimation 方法生成概率密度函数
kde_model = KernelDensity(kernel='gaussian', bandwidth=100)
kde_model.fit(np.array(point).reshape(-1, 1))
density2 = np.exp(kde_model.score_samples(quantiles_res.reshape(-1, 1)))


# 可视化展示概率密度函数、实际值和模型预测值
fig , ax = plt.subplots(figsize=(12, 6))
ax.plot(quantiles_res, density0, label='gauss')
ax.plot(quantiles_res, density1, label='gauss mixture')
ax.plot(quantiles_res, density2, label='kernel density estimation')
y_test = scaler_y.inverse_transform(y_test)
ax.vlines(y_test[idx], 0, max(density0.max(), density1.max(), density2.max())*1.2, label='true value', color='r')
plt.xlabel('Carbon Emissions')
plt.ylabel('Probability Density')
plt.legend()
plt.show()