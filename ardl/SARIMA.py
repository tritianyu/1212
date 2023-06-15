import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\造纸及纸制品业.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]

mod = sm.tsa.statespace.SARIMAX(endog=y_train,
                                exog=x_train,
                                order=[3,2,2],
                                seasonal_order=[0,0,0,12],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
result = mod.fit()
train_forecast = result.fittedvalues
train_forecast = pd.Series(train_forecast, index=y_train.index, name='train_forecast')
forecast = result.forecast(steps=len(y_test), exog=x_test)
forecast = pd.Series(forecast, index=y_test.index, name='forecast')
err=mean_absolute_percentage_error(forecast,y_test)
print(err)

pred1=result.get_prediction()
pred1_ci=pred1.conf_int()
pred2=result.get_forecast(steps=len(y_test),exog=x_test)
pred2_ci=pred2.conf_int()
ax =y_data.plot(figsize=(20, 16))
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('碳排放量')
plt.xlabel('SARIMA')
plt.legend()
plt.show()
