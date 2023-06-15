import itertools

import pandas as pd
import statsmodels.tsa.ardl as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电力、热力的生产和供应业月份数据.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]

x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)
x_train=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)
# 定义参数搜索范围
lags_range = range(1, 6)
order_range = range(1, 6)

# 生成所有可能的参数组合
param_list = list(itertools.product(lags_range,order_range))

# 遍历所有参数组合，计算BIC值
min_bic = float('inf')
best_params = None
for params in param_list:
    try:
        model = sm.ARDL(endog=y_train, exog=x_train, lags=params[0], order=params[1])
        result = model.fit()
        if result.bic < min_bic:
            min_bic = result.bic
            best_params = params
    except:
        pass

# 输出BIC值最小的模型对应的参数
print('Best parameters:', best_params)
print('Minimum BIC:', min_bic)

