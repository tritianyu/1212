from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SVRQuantileRegressor:
    def __init__(self, q=0.5, C=1.0, epsilon=0.1, kernel='rbf'):
        self.q = q
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.models = []

    def fit(self, X, y):
        y_lower = y.copy()  # 训练下分位数模型
        y_lower[y_lower >= np.percentile(y_lower, self.q*100)] = 1
        y_lower[y_lower < np.percentile(y_lower, self.q*100)] = 0
        model_lower = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        model_lower.fit(X, y_lower)
        self.models.append(model_lower)

        y_upper = y.copy()  # 训练上分位数模型
        y_upper[y_upper <= np.percentile(y_upper, self.q*100)] = 0
        y_upper[y_upper > np.percentile(y_upper, self.q*100)] = 1
        model_upper = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
        model_upper.fit(X, y_upper)
        self.models.append(model_upper)

    def predict(self, X):
        predictions = []
        for i in range(len(self.models)):
            model = self.models[i]
            pred = model.predict(X)
            predictions.append(pred)
        lower, upper = predictions[0], predictions[1]
        y_pred = lower * (1 - self.q) + upper * self.q
        return y_pred


if __name__ == '__main__':
    data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电气机械及器材制造业.xlsx")
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
    model = SVRQuantileRegressor(q=0.99)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    y_pred = y_pred.reshape(-1,1)
    y_train_pred = y_train_pred.reshape(-1, 1)
    y_test = scaler_y.inverse_transform(y_test)
    y_train = scaler_y.inverse_transform(y_train)
    y_pred = scaler_y.inverse_transform(y_pred)
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