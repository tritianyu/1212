import pandas as pd
import statsmodels.tsa.holtwinters as hw
import matplotlib.pyplot as plt
import numpy as np

# 读取 Excel 表格中的数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\黑色金属冶炼及压延加工业.xlsx", index_col=0)
data=data[["电力（万千瓦时）","碳排放"]]
# 重命名列名
new_columns = {
    "电力（万千瓦时）": "x",
    "碳排放": "y"
}
data = data.rename(columns=new_columns)
data.index = pd.to_datetime(data.index)
# 将数据按照月份重采样
monthly_data = data.resample("M").sum()

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
y_train, y_test = monthly_data['y'][:train_size], monthly_data['y'][train_size:]
x_train, x_test = monthly_data['x'][:train_size], monthly_data['x'][train_size:]
# 训练 Holt-Winters 季节性模型
model = hw.ExponentialSmoothing(y_train.values, seasonal_periods=12, trend="add", seasonal="add")
result = model.fit()

# 对测试集进行预测
forecast = result.forecast(steps=len(y_test))

# 将预测结果转换成 Pandas 数据帧
forecast_df = pd.DataFrame(index=x_test.index, data=forecast, columns=['Forecast'])

# 计算 MSE 评估模型
mse = np.mean((forecast - y_test.values) ** 2)
print(f"MSE: {mse:.2f}")
# 绘制预测结果图像
plt.figure(figsize=(12,6))
plt.plot(x_train.index, y_train, label="Train")
plt.plot(x_test.index, y_test, label="Test")
plt.plot(x_test.index, forecast, label="Forecast")

# 计算置信区间
forecast_lower = forecast - 1.96 * np.std(result.resid)
forecast_upper = forecast + 1.96 * np.std(result.resid)
plt.fill_between(x_test.index, forecast_lower, forecast_upper, alpha=0.2, label="95% Confidence Interval")

plt.legend(loc="upper left")
plt.title("Electricity and Carbon Emission Prediction")
plt.xlabel("Year")
plt.ylabel("Emission")
plt.show()