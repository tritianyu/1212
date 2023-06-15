import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity
from  sklearn.metrics import mean_absolute_percentage_error
# 加载电力数据和碳排放数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电力、热力的生产和供应业月份数据.xlsx")
x = data["电力（万千瓦时）"].values.reshape(-1,1)
y = data["碳排放"].values.reshape(-1,1)

# 分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 训练模型并进行预测
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(x_train, y_train.ravel())
y_train_pred = rf.predict(x_train)
y_test_pred = rf.predict(x_test)
y_pred = rf.predict(np.concatenate([x_train, x_test]))
err=mean_absolute_percentage_error(y_test,y_test_pred)
print(err)
# 计算预测置信区间
quantiles = np.arange(0.01, 1, 0.01)
quantiles_res = np.quantile(y_pred, quantiles, axis=0)

# 采用 kernel density estimation 方法生成概率密度函数
kde_model = KernelDensity(kernel='gaussian', bandwidth=1.0)
kde_model.fit(y_pred.reshape(-1,1))
densities = np.exp(kde_model.score_samples(quantiles_res.reshape(-1,1)))

# 获取实际值所在位置的概率密度，并将值绘制在图上
true_value_density = np.exp(kde_model.score_samples(y_test))
true_value_x = np.arange(0, true_value_density.size)
true_value_y = true_value_density.ravel()
# 可视化展示概率密度函数、实际值和模型预测值
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.arange(0, y_train.size), y_train.ravel(), 'r', label='true')
ax.plot(np.arange(0, y_train.size), y_train_pred, 'g', label='train prediction')
ax.plot(np.arange(y_train.size, y_pred.size), y_test.ravel(), 'r')
ax.plot(np.arange(y_train.size, y_pred.size), y_test_pred, 'b', label='test prediction')

# 绘制真实值的散点图

plt.xlabel('Indices')
plt.ylabel('Carbon Emissions')
plt.legend()
plt.show()