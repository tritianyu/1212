import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow.python.keras.layers import Dense
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电力、热力的生产和供应业月份数据.xlsx")
x_data=data["电力（万千瓦时）"]
y_data=data["碳排放"]
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=33)
y_picture=y_train
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train.values.reshape(-1,1))
x_test = ss_x.transform(x_test.values.reshape(-1,1))

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.values.reshape(-1,1))
y_test = ss_y.transform(y_test.values.reshape(-1,1))
model = keras.models.Sequential()
model.add(Dense(50, input_dim=1, use_bias=True, kernel_initializer='RandomNormal', activation='relu',))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1,))
model.compile(optimizer='adam', loss=keras.losses.MeanAbsoluteError(),metrics=['mape'])
model.fit(x_train, y_train,  epochs=1000, verbose=1)
predict_train = (ss_y.inverse_transform(model.predict(x_train))).ravel()
actual_train = (ss_y.inverse_transform(y_train)).ravel()
train_mae = mean_absolute_error(actual_train, predict_train)


predict_test = (ss_y.inverse_transform(model.predict(x_test))).ravel()
actual_test = (ss_y.inverse_transform(y_test)).ravel()
test_mae = mean_absolute_error(actual_test, predict_test)
history = model.fit(x_train, y_train,batch_size=512,
                    epochs=1000, validation_split = 0.3, verbose=0)
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label = 'Val Error')
    plt.legend()
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.plot(hist['epoch'], hist['mape'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mape'],
             label = 'Val Error')
    plt.legend()
    plt.show()


plot_history(history)
plt.scatter(x_train,predict_train,c='r')
plt.scatter(x_train,y_picture,c='b')
plt.ylabel("C(tCO2)")
plt.xlabel("elec(kwh)")
plt.show()
print(sum(predict_train)-sum(y_picture))
