import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from main import RBFQuantileRegressor
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings(action='ignore')
# 加载电力数据和碳排放数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电气机械及器材制造业.xlsx")
x = data["电力（万千瓦时）"].values.reshape(-1,1)
y = data["碳排放"].values.reshape(-1,1)

# 分割训练集和测试集
train_size = int(len(data) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]
x_train, x_test = x[:train_size], x[train_size:]
x_train=sm.add_constant(x_train)
x_test=sm.add_constant(x_test)
# 定义预测分位数并训练模型
quantiles = np.arange(0.01, 1, 0.01)
predictions = []
for quantile in quantiles:
    rbf=RBFQuantileRegressor(n_units=5,q=quantile,gamma=0.1,max_iter=500,learning_rate=0.1)
    result=rbf.fit(x_train,y_train)
    y_pred=rbf.predict(x_test)
    predictions.append(y_pred)
# 绘制某个数据点的概率密度
idx = 57# 假设要查看第个样本点对应的概率密度
point =[p[idx] for p in predictions]
quantiles_res = np.quantile(point, quantiles)
loc = np.mean(point)
scale = np.std(point)
point.sort()
#采用正态分布
#density = norm.pdf(point, loc=loc, scale=scale)

#采用 kernel density estimation 方法生成概率密度函数
kde_model = KernelDensity(kernel='gaussian', bandwidth=100)
kde_model.fit(np.array(point).reshape(-1,1))
density = np.exp(kde_model.score_samples(quantiles_res.reshape(-1,1)))

# 可视化展示概率密度函数、实际值和模型预测值
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(point, density, label='probability density function')
ax.vlines(y_test[idx],0,density.max()*1.2, label='true value', color='r')
plt.xlabel('Carbon Emissions')
plt.ylabel('Probability Density')
plt.legend()
plt.show()