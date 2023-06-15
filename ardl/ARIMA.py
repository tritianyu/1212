import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电力、热力的生产和供应业月份数据.xlsx")
x_data = data["电力（万千瓦时）"]
y_data = data["碳排放"]

train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]

model = ARIMA(endog=y_train, exog=x_train, order=(3,2,2))
result = model.fit()

train_forecast = result.predict(start=0, end=train_size-1)
train_forecast = pd.Series(train_forecast, index=y_train.index, name='train_forecast')

forecast = result.predict(start=train_size, end=len(y_data)-1, exog=x_test)
forecast = pd.Series(forecast, index=y_test.index, name='forecast')

error = mean_absolute_percentage_error(y_test, forecast)
print(error)
pred1=result.get_prediction()
pred1_ci=pred1.conf_int()
pred2=result.get_forecast(steps=len(y_test),exog=x_test)
pred2_ci=pred2.conf_int()
in_conf_int = 0
for i in range(len(y_test)):
    if (y_test.iloc[i].item() >= pred2_ci.iloc[i,0]) and (y_test.iloc[i].item() <= pred2_ci.iloc[i,1]):
        in_conf_int += 1
picp = in_conf_int / len(y_test)
print("预测区间覆盖率(PICP):", picp)

ax =y_data.plot(figsize=(20, 16))
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('碳排放量')
plt.xlabel('ARIMA')
plt.legend()
plt.show()