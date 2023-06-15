import warnings
warnings.filterwarnings(action='ignore')
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]
q = d = range(0, 2)
p = range(0, 4)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(endog=y_train,
                                            exog=x_train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()


            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue
print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))
'''
mod = sm.tsa.statespace.SARIMAX(endog=y_train,
                                exog=x_train,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
result = mod.fit()
print(result.summary())

train_forecast = result.fittedvalues
train_forecast = pd.Series(train_forecast, index=y_train.index, name='train_forecast')
forecast = result.forecast(steps=len(y_test), exog=x_test)
forecast = pd.Series(forecast, index=y_test.index, name='forecast')
pred1=result.get_prediction()
pred1_ci=pred1.conf_int()
pred2=result.get_forecast(steps=len(y_test),exog=x_test)
pred2_ci=pred2.conf_int()
ax =y_data.plot(figsize=(20, 16))
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('碳排放量')
plt.xlabel('SARIMAX')
plt.legend()
plt.show()
'''
