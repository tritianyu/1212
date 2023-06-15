import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# 读取Excel文件并提取需要计算相关性的列数据
data = pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电气机械及器材制造业.xlsx")
data = data[["原煤", "汽油", "煤油", "柴油", "燃料油", "热力", "电力（万千瓦时）"]]

# 计算相关系数矩阵
corr = np.corrcoef(data.T)

# 绘制热力图
sns.set(font_scale=1.1)
sns.heatmap(corr, annot=True, cmap="coolwarm",
            xticklabels=data.columns.values,
            yticklabels=data.columns.values)
plt.xticks(rotation=45)
plt.show()