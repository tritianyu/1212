import pandas as pd
import statsmodels.tsa.ardl as sm
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\造纸及纸制品业.xlsx")
x_data=data[["电力（万千瓦时）","原煤"]]
y_data=data["碳排放"]
train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]
x_data=pd.DataFrame(x_data)
x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)
x_train=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)
model = sm.ARDL(endog=y_train, exog=x_train, lags=5,order=2)
result = model.fit()
#print(result.summary())
forecast = result.forecast(steps=len(y_test), exog=x_test)
forecast = pd.Series(forecast, index=y_test.index, name='forecast')
err=mean_absolute_percentage_error(y_test,forecast)
print(err)

pred1=result.get_prediction()
pred1_ci=pred1.conf_int()
pred2=result.get_prediction(start=len(x_train),end=len(x_train)+len(x_test)-1,exog_oos=x_test)
pred2_ci=pred2.conf_int()
# 计算预测区间覆盖率
in_conf_int = 0
for i in range(len(y_test)):
    if (y_test.iloc[i].item() >= pred2_ci.iloc[i,0]) and (y_test.iloc[i].item() <= pred2_ci.iloc[i,1]):
        in_conf_int += 1
picp = in_conf_int / len(y_test)
print("预测区间覆盖率(PICP):", picp)

ax =y_data.plot(figsize=(20, 16))
pred1.predicted_mean.plot(ax=ax, label='测试集预测结果')
pred2.predicted_mean.plot(ax=ax, label='训练集预测结果')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('碳排放量')
plt.xlabel('ARDL')
plt.legend()
plt.show()