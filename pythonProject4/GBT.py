
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd


class GBTQuantileRegressor:
    def __init__(self, q=0.5, n_estimators=100):
        self.q = q
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        self.model = GradientBoostingRegressor(loss='quantile', alpha=self.q, n_estimators=self.n_estimators,
                                              max_depth=1, learning_rate=0.1)
        self.model.fit(X, y)


    def predict(self, X):
        a = self.model.predict(X)
        return a


if __name__ == '__main__':
    data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\化学原料及化学制品制造业.xlsx")
    x_data = data["电力（万千瓦时）"]
    y_data = data["碳排放"]

    train_size = int(len(data) * 0.8)
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
    model = GBTQuantileRegressor(q=0.98,n_estimators=200)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    y_pred=y_pred.reshape(-1,1)
    y_train_pred = y_train_pred.reshape(-1, 1)
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



