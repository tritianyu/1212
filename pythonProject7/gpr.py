import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\黑色金属冶炼及压延加工业.xlsx")

# 数据清洗和处理
data = data.dropna()
X = data['电力（万千瓦时）'].values.reshape(-1, 1)
y = data['碳排放'].values

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立高斯过程回归模型
kernel = C() * RBF() + WhiteKernel()
model = GaussianProcessRegressor(kernel=kernel)
model.fit(X_train, y_train)
# 预测
x_pred = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred, sigma = model.predict(x_pred, return_std=True)

# 可视化结果
plt.plot(X_train, y_train, color='blue', alpha=0.5, label='train data')
plt.plot(X_test, y_test, color='green', alpha=0.5, label='test data')
plt.plot(x_pred, y_pred, color='red', label='prediction')
plt.fill_between(x_pred.flatten(), y_pred - 2 * sigma, y_pred + 2 * sigma,
                 color='gray', alpha=0.2, label='95% confidence interval')
plt.legend(loc='upper left')
plt.xlabel('Time')
plt.ylabel('Carbon')
plt.show()