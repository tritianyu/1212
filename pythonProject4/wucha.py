import numpy as np
import pandas as pd
from GBT import GBTQuantileRegressor
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
y_train = y_train.ravel()
y_test = scaler_y.fit_transform(y_test)
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
predictions = predictions.reshape(len(x_test),len(quantiles))

y_test = scaler_y.inverse_transform(y_test)
density_list = []
density_real_list = []

for i in range(len(predictions)):
    kde_model = KernelDensity(kernel='gaussian', bandwidth=500)
    kde_model.fit(predictions[i].reshape(-1, 1))
    density = np.exp(kde_model.score_samples(predictions[i].reshape(-1, 1)))
    density_real = np.exp(kde_model.score_samples(y_test[i].reshape(-1, 1)))
    density_list.append(density)
    density_real_list.append(density_real)
'''
for i in range(len(predictions)):
    gmm = GaussianMixture(n_components=3, max_iter=200,random_state=1)
    gmm.fit(predictions[i].reshape(-1, 1))
    density = np.exp(gmm.score_samples(predictions[i].reshape(-1, 1)))
    density_list.append(density)
    density_real = np.exp(gmm.score_samples(y_test[i].reshape(-1, 1)))
    density_real_list.append(density_real)
'''
density_list=np.vstack(density_list)
density_real_list = np.vstack(density_real_list)
print(density_real_list)


def Theil():
    err0 = 0
    for i in range(len(predictions)):
        x_min , x_max = np.min(predictions[i]), np.max(predictions[i])
        step = (x_max - x_min) / len(predictions[i])
        p1 = max(density_list[i])*step
        p2 = density_real_list[i]*step
        err0 = err0 + p2*abs(np.log(p2/p1))
        print(p1, p2)
    return err0 / len(predictions)


e = Theil()
print(e)
