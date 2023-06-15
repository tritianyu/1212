import pandas as pd
import statsmodels.api as sa
import statsmodels.tsa.ardl as sm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn.metrics import mean_squared_log_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电气机械及器材制造业.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]

x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)
x_train=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)

#ardl
model0 = sm.ARDL(endog=y_train, exog=x_train, lags=5,order=2)
result0 = model0.fit()
train_forecast0 = result0.fittedvalues
train_forecast0 = pd.Series(train_forecast0, index=y_train.index, name='train_forecast')
forecast0 = result0.forecast(steps=len(y_test), exog=x_test)
forecast0 = pd.Series(forecast0, index=y_test.index, name='forecast')
err0=mean_squared_log_error(y_test,forecast0)


#arima
model1 = ARIMA(endog=y_train, exog=x_train, order=(3,2,2))
result1 = model1.fit()

train_forecast1 = result1.predict(start=0, end=train_size-1)
train_forecast1 = pd.Series(train_forecast1, index=y_train.index, name='train_forecast')

forecast1 = result1.predict(start=train_size, end=len(y_data)-1, exog=x_test)
forecast1 = pd.Series(forecast1, index=y_test.index, name='forecast')

err1 = mean_squared_log_error(y_test, forecast1)

#sarimax
model2 = sa.tsa.statespace.SARIMAX(endog=y_train,
                                exog=x_train,
                                order=[3,2,2],
                                seasonal_order=[0,0,0,12],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
result2 = model2.fit()
train_forecast2 = result2.fittedvalues
train_forecast2 = pd.Series(train_forecast2, index=y_train.index, name='train_forecast')
forecast2 = result2.forecast(steps=len(y_test), exog=x_test)
forecast2 = pd.Series(forecast2, index=y_test.index, name='forecast')
err2=mean_squared_log_error(forecast2,y_test)
print(err0)
print(err1)
print(err2)





fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test, label='碳排放实际值')
ax.plot(forecast0, label='ARDL预测结果')
ax.plot(forecast1, label='ARIMA预测结果')
ax.plot(forecast2, label='SARIMAX预测结果')
ax.legend()
plt.show()

