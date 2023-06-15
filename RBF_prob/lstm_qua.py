import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import load_model
from keras.utils import custom_object_scope
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

class LSTMQuantileRegressor():
    def __init__(self,q=0.5,n_units=32):
        self.q=q
        self.n_units=n_units
        self.model=Sequential()

    def quantile_loss(self,y_true, y_pred):
        error = y_true - y_pred
        return tf.keras.backend.mean(tf.keras.backend.maximum((self.q - 1) * error,self.q * error), axis=-1)
 
    def fit(self,x_train,y_train,x_test,y_test):
        self.model.add(LSTM(units=self.n_units*2,return_sequences=True,input_shape=(x_train.shape[1],1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.n_units,return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.n_units))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        with custom_object_scope({'quantile_loss':self.quantile_loss}):
            self.model.compile(optimizer='adam',loss='quantile_loss')
            es=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)
            mc=ModelCheckpoint(r'C:\Users\14684\PycharmProjects\RBF_prob\model_checkpoint.h5', monitor='val_loss', mode='min', save_best_only=True)
            history=self.model.fit(x_train,y_train,epochs=500,batch_size=32,validation_data=(x_test,y_test),callbacks=[es,mc])

    def predict(self,x_test):
        with custom_object_scope({'quantile_loss':self.quantile_loss}):
            best_model=load_model(r'C:\Users\14684\PycharmProjects\RBF_prob\model_checkpoint.h5')
        x_pred=best_model.predict(x_test)
        return x_pred


if __name__ == '__main__':
    data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
    x_data = data["电力（万千瓦时）"]
    y_data = data["碳排放"]

    train_size = int(len(data) * 0.7)
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.fit_transform(x_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.fit_transform(y_test)
    model = LSTMQuantileRegressor(n_units=32, q=1)
    model.fit(x_train, y_train,x_test,y_test)

    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)

    y_test=scaler_y.inverse_transform(y_test)
    y_train=scaler_y.inverse_transform(y_train)
    y_pred=scaler_y.inverse_transform(y_pred)
    y_train_pred = scaler_y.inverse_transform(y_train_pred)
    # 绘制训练集的预测结果和真实结果
    plt.subplot(2, 1, 1)
    plt.plot(y_train, label='Real')
    plt.plot(y_train_pred, label='Predicted')
    plt.title('Training Set Predictions')
    plt.ylabel('Value')
    plt.legend()

    # 绘制测试集的预测结果和真实结果
    plt.subplot(2, 1, 2)
    plt.plot(y_test, label='Real')
    plt.plot(y_pred, label='Predicted')
    plt.title('Test Set Predictions')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    '''
    # 绘制y_train、y_pred、y_test的曲线图
    plt.plot(y_data, label='y_train')
    plt.plot(y_train_pred, label='y_pred')
    plt.plot(range(len(x_train), len(x_train) + len(x_test)), y_pred, label='y_test')
    plt.legend()
    plt.show()'''



