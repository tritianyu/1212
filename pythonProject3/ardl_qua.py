import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.ardl as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_percentage_error
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]

x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)
x_train=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)
model = sm.ARDL(endog=y_train, exog=x_train, lags=5,order=2)
result = model.fit()
train_forecast = result.fittedvalues
train_forecast = pd.Series(train_forecast, index=y_train.index, name='train_forecast')
forecast = result.forecast(steps=len(y_test), exog=x_test)
forecast = pd.Series(forecast, index=y_test.index, name='forecast')


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_data, label='y')
ax.plot(train_forecast, label='train_forecast')
ax.plot(forecast, label='forecast')
ax.legend()
plt.show()