import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings(action='ignore')
class RBFQuantileRegressor():
    def __init__(self, q=0.1, gamma=0.1, n_units=10, learning_rate=0.1, max_iter=500):
        self.q = q
        self.gamma = gamma
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._hidden_layer = MLPRegressor(hidden_layer_sizes=(n_units,), activation='tanh', solver='lbfgs', learning_rate_init=learning_rate, max_iter=max_iter,random_state=1)
        self._centers = None
        self._weights = None

    def _rbf(self, x, c):
        return np.exp(-self.gamma * np.linalg.norm(x - c) ** 2)

    def _rbf_matrix(self, X):
        return np.array([[self._rbf(x, c) for c in self._centers] for x in X])

    def _quantile_loss(self, y_true, y_pred):
        error = y_true - y_pred
        return np.maximum((self.q - 1) * error, self.q * error)

    def fit(self, X_train, y_train):

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        kmeans = KMeans(n_clusters=self.n_units)
        kmeans.fit(X_train_scaled)
        self._centers = kmeans.cluster_centers_
        hidden_train = self._rbf_matrix(X_train_scaled)
        self._hidden_layer = MLPRegressor(hidden_layer_sizes=(self.n_units * 5,), activation='relu', solver='lbfgs',
                                          learning_rate_init=self.learning_rate, max_iter=self.max_iter,random_state=1)
        self._hidden_layer.fit(hidden_train, y_train)
        hidden_train = self._hidden_layer.predict(hidden_train).reshape((-1, 1))
        self._weights = np.linalg.inv(hidden_train.T.dot(hidden_train)).dot(hidden_train.T).dot(y_train)
        '''
        自适应调整gamma
        k=10
        X_train_dist=np.abs(X_train_scaled[:,np.newaxis]-X_train_scaled)
        X_train_dist=np.sort(X_train_dist,axis=1)[:,1:k+1]
        gamma_tmp=1 / (k * np.mean(X_train_dist, axis=1))
        self.gamma=np.mean(gamma_tmp)
        '''
    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        hidden_test = self._rbf_matrix(X_test_scaled)
        hidden_test = self._hidden_layer.predict(hidden_test).reshape((-1, 1))
        return hidden_test.dot(self._weights)
if __name__=='__main__':
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
    model = RBFQuantileRegressor(n_units=5, q=2, gamma=0.1, max_iter=500,learning_rate=0.1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    err=mean_absolute_percentage_error(y_pred,y_test)
    print(err)
    # 绘制训练结果
    y_train_pred = model.predict(x_train)
    # 绘制y_train、y_pred、y_test的曲线图
    plt.plot(y_data, label='y_train')
    plt.plot(y_train_pred, label='y_pred')
    plt.plot(range(len(x_train), len(x_train) + len(x_test)), y_pred, label='y_test')
    plt.legend()
    plt.show()

