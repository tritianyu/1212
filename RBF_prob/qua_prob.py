import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings(action='ignore')
# 加载电力数据和碳排放数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\食品制造业.xlsx")
x = data["电力（万千瓦时）"].values.reshape(-1,1)
y = data["碳排放"].values.reshape(-1,1)

# 分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train=sm.add_constant(x_train)
x_test=sm.add_constant(x_test)
# 定义预测分位数并训练模型
quantiles = np.arange(0.01, 1, 0.1)
predictions = []
for quantile in quantiles:
    rbf=sm.QuantReg(y_train,x_train)
    result=rbf.fit(q=quantile)
    y_pred=result.predict(x_test)
    predictions.append(y_pred)
predictions=np.vstack(predictions).T
print(predictions.shape)
density_list = []
#gauss
for i in range(len(predictions)):
    kde_model_i = KernelDensity(bandwidth=100, kernel='gaussian')
    kde_model_i.fit(predictions[i].reshape(-1, 1))
    density=np.exp(kde_model_i.score_samples(predictions[i].reshape(-1,1)))
    density_list.append(density)
print(density_list)
x_min,x_max=np.min(predictions[0]),np.max(predictions[0])
step = (x_max - x_min)/len(predictions[0])
probability = density_list[0]*step
probability=probability/sum(probability)
real=np.ones(10)/10
err=np.linalg.norm(real-probability)
print(err)
'''
# 绘制某个数据点的概率密度
idx =0# 假设要查看第个样本点对应的概率密度
point =[p[idx] for p in predictions]
quantiles_res = np.quantile(point, quantiles)
loc = np.mean(point)
scale = np.std(point)
point.sort()
#采用正态分布
#point.sort()
#density = norm.pdf(point, loc=loc, scale=scale)

#采用 kernel density estimation 方法生成概率密度函数
kde_model = KernelDensity(kernel='gaussian', bandwidth=100)
kde_model.fit(np.array(point).reshape(-1,1))
density = np.exp(kde_model.score_samples(quantiles_res.reshape(-1,1)))

# 可视化展示概率密度函数、实际值和模型预测值


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(point, density, label='probability density function')
ax.vlines(y_test[idx], 0,density.max()*1.2, label='true value', color='r')
ax.vlines(sum(point)/len(point),0, density.max()*1.2, label='predicted value', color='b')
plt.xlabel('Carbon Emissions')
plt.ylabel('Probability Density')
plt.legend()
plt.show()'''
