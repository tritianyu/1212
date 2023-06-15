import tensorflow
import matplotlib.pyplot as plt
from tensorflow.python import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.layers import Dense
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
#x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=123)
train_size = int(len(data) * 0.8)
y_train, y_test = y_data[:train_size], y_data[train_size:]
x_train, x_test = x_data[:train_size], x_data[train_size:]
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train.values.reshape(-1,1))
x_test = ss_x.transform(x_test.values.reshape(-1,1))

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.values.reshape(-1,1))
y_test = ss_y.transform(y_test.values.reshape(-1,1))

model = keras.models.Sequential()
model.add(Dense(100, input_dim=1, use_bias=True, kernel_initializer='TruncatedNormal', activation='relu',))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,))
model.compile(optimizer='adam', loss=keras.losses.MeanAbsoluteError(),metrics=['mape'])
model.fit(x_train, y_train,  epochs=1000, verbose=1,batch_size=16)
predict_train = (ss_y.inverse_transform(model.predict(x_train)))
actual_train = (ss_y.inverse_transform(y_train))
train_mae = mean_absolute_error(actual_train, predict_train)
predict_test = (ss_y.inverse_transform(model.predict(x_test)))
actual_test=(ss_y.inverse_transform(y_test))
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(actual_train,label='actual_train',c='C0')
ax.plot(predict_train,label='predict_train',c='r')
ax.plot(range(len(actual_train),len(actual_train)+len(actual_test)),actual_test,label='actual_test',c='C0')
ax.plot(range(len(predict_train),len(predict_train)+len(predict_test)),predict_test, label='predict_test',c='g')
ax.legend
err=mean_absolute_error(predict_train,actual_train)
plt.show()

