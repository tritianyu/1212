import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
# 读取数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")

# 筛选特征
x_data = data["电力（万千瓦时）"].values.reshape(-1, 1)
y_data = data["碳排放"].values.reshape(-1, 1)

# 划分数据集
train_size = int(len(data) * 0.7)
train_x, test_x = x_data[:train_size, :], x_data[train_size:, :]
train_y, test_y = y_data[:train_size, :], y_data[train_size:, :]

# 数据预处理
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
train_x = scaler_x.fit_transform(train_x)
test_x = scaler_x.transform(test_x)
train_y = scaler_y.fit_transform(train_y)
test_y = scaler_y.transform(test_y)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# 构建EarlyStopping和ModelCheckpoint回调函数
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint(r'C:\Users\14684\PycharmProjects\tensorflow\model_checkpoint.h5', monitor='val_loss', mode='min', save_best_only=True)

# 训练模型
history = model.fit(train_x, train_y, epochs=500, batch_size=32, validation_data=(test_x, test_y), callbacks=[es, mc])

# 绘制学习曲线

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.show()

# 加载最优模型并进行预测
best_model = load_model(r'C:\Users\14684\PycharmProjects\tensorflow\model_checkpoint.h5')
pred_train = best_model.predict(train_x)  # 预测训练集的输出结果
pred_test = best_model.predict(test_x)  # 预测测试集的输出结果
err=mean_absolute_error(test_y,pred_test)
print(err)
# 还原预测结果
real_prediction_train = scaler_y.inverse_transform(pred_train)
real_y_train = scaler_y.inverse_transform(train_y)

real_prediction_test = scaler_y.inverse_transform(pred_test)
real_y_test = scaler_y.inverse_transform(test_y)
# 绘制预测结果图
plt.figure(figsize=(16, 8))
model.summary()
# 绘制训练集的预测结果和真实结果
plt.subplot(2, 1, 1)
plt.plot(real_y_train, label='Real')
plt.plot(real_prediction_train, label='Predicted')
plt.title('Training Set Predictions')
plt.ylabel('Value')
plt.legend()

# 绘制测试集的预测结果和真实结果
plt.subplot(2, 1, 2)
plt.plot(real_y_test, label='Real')
plt.plot(real_prediction_test, label='Predicted')
plt.title('Test Set Predictions')
plt.ylabel('Value')
plt.legend()
plt.show()
