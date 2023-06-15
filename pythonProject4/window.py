from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


class SVRQuantileRegressor:
    def __init__(self, q=0.5, C=1.0, epsilon=0.1, kernel='rbf'):
        self.q = q
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.models = []

    def fit(self, X, y):
        y_lower = y.copy()  # 训练下分位数模型
        y_lower[y_lower >= np.percentile(y_lower, self.q*100)] = 1
        y_lower[y_lower < np.percentile(y_lower, self.q*100)] = 0
        model_lower = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        model_lower.fit(X, y_lower)
        self.models.append(model_lower)

        y_upper = y.copy()  # 训练上分位数模型
        y_upper[y_upper <= np.percentile(y_upper, self.q*100)] = 0
        y_upper[y_upper > np.percentile(y_upper, self.q*100)] = 1
        model_upper = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        model_upper.fit(X, y_upper)
        self.models.append(model_upper)

    def predict(self, X):
        predictions = []
        for i in range(len(self.models)):
            model = self.models[i]
            pred = model.predict(X)
            predictions.append(pred)
        lower, upper = predictions[0], predictions[1]
        y_pred = lower * (1 - self.q) + upper * self.q
        return y_pred



def open_file():
    global data, x_data, y_data
    # 弹出文件选择对话框，获取文件路径
    path = filedialog.askopenfilename()
    if path:
        # 读取数据文件
        data = pd.read_excel(path)
        x_data = data["电力（万千瓦时）"]
        y_data = data["碳排放"]
        # 在这里处理读取的数据，可以调用其他函数进行进一步的操作
        print("读取成功！")
        window.destroy()

    else:
        print("没有选择文件！")

def split():
    global data, x_data, y_data
    train_size = int(len(data) * 0.8)
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


def get_prediction():
    quantiles = np.arange(0.01, 1, 0.01)
    predictions = []
    for quantile in quantiles:
        model = SVRQuantileRegressor(q=quantile)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = y_pred.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred)
        predictions.append(y_pred)
    predictions = np.dstack(predictions)
    predictions = predictions.reshape(len(x_test), len(quantiles))
    return predictions
# 创建主窗口
window = tk.Tk()

# 添加一个标签和一个按钮
label = tk.Label(window, text="请输入数据文件地址：")
label.pack()

button = tk.Button(window, text="选择文件", command=open_file)
button.pack()

# 启动窗口的消息循环
window.mainloop()
x_train, x_test, y_train, y_test = split()
#标准化
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.fit_transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_train = y_train.ravel()
y_test = scaler_y.fit_transform(y_test)

predictions = get_prediction()

idx = 32 # 假设要查看第个样本点对应的概率密度
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
# 定义带宽参数的范围
bandwidth_range = np.linspace(100, 1000, 100)

# 创建KernelDensity对象
kde_model = KernelDensity(kernel='gaussian')

# 创建GridSearchCV对象
grid_search = GridSearchCV(kde_model, {'bandwidth': bandwidth_range}, cv=5)

# 调用fit方法进行带宽参数的搜索
grid_search.fit(np.array(point).reshape(-1, 1))

# 输出最优的带宽参数
best_bandwidth = grid_search.best_params_['bandwidth']
print("最优带宽参数:", best_bandwidth)

# 使用最优带宽参数重新创建KernelDensity对象
kde_model_best = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth)

# 使用最优带宽参数拟合数据并计算概率密度
kde_model_best.fit(np.array(point).reshape(-1, 1))
density2 = np.exp(kde_model_best.score_samples(quantiles_res.reshape(-1, 1)))
'''
kde_model = KernelDensity(kernel='gaussian', bandwidth=100)
kde_model.fit(np.array(point).reshape(-1, 1))
density2 = np.exp(kde_model.score_samples(quantiles_res.reshape(-1, 1)))
'''

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