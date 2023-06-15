import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# 读取电力和碳排放的数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电气机械及器材制造业.xlsx")

# 筛选出需要的列
data = data[['电力（万千瓦时）', '碳排放']]

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# K-means 聚类分析
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data_scaled)
score = silhouette_score(data_scaled, kmeans.labels_)
# 输出结果
print('平均轮廓系数：%.2f' % score)
# 输出聚类结果
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans.labels_)
plt.xlabel('Electricity')
plt.ylabel('CO2 Emissions')
plt.show()