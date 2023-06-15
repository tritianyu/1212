import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings(action='ignore')

# 读入数据
data = pd.read_excel(r"C:\Users\14684\Desktop\stirpat\电.xlsx")
dep= pd.read_excel(r"C:\Users\14684\Desktop\stirpat\碳排放.xlsx")
pred=pd.read_excel(r"C:\Users\14684\Desktop\stirpat\pred.xlsx")
# 定义变量
dependent = dep["发电及供热"]
population = data['population']
affluence = data['affluence']
urbanization = data['urbanization']
m=data['m']
n=data['n']
dependent=dependent.values.tolist()
population=population.values.tolist()
affluence=affluence.values.tolist()
urbanization=urbanization.values.tolist()
m=m.values.tolist()
n=n.values.tolist()
#取对数
for i in range(len(dependent)):
    dependent[i]=math.log(dependent[i])
for i in range(len(population)):
    population[i]=math.log(population[i])
for i in range(len(urbanization)):
    urbanization[i]=math.log(urbanization[i])
for i in range(len(affluence)):
    affluence[i]=math.log(affluence[i])
for i in range(len(m)):
    m[i]=math.log(m[i])
for i in range(len(n)):
    n[i]=math.log(n[i])
# 定义模型
dependent=pd.DataFrame(dependent)
population=pd.DataFrame(population)
affluence=pd.DataFrame(affluence)
urbanization=pd.DataFrame(urbanization)
m=pd.DataFrame(m)
n=pd.DataFrame(n)
X = sm.add_constant(pd.concat([population, affluence, urbanization,m,n], axis=1))
y = dependent
# 计算岭回归系数
alphas = np.logspace(-7,0, 100)
ridge=Ridge()
# 用列表 coefs 记录不同 alpha 下模型的系数
coefs = np.zeros((100, X.shape[1]))
for i, a in enumerate(alphas):
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs[i, :] = ridge.coef_.ravel()

# 绘制 alpha 和每个特征系数的关系图
plt.figure(figsize=(10, 6))
ax = plt.gca()
for i in range(X.shape[1]):
    ax.plot(alphas, coefs[:, i],label=data.feature_names[i])
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('Alpha', fontsize=12)
plt.ylabel('Coefficients', fontsize=12)
plt.title('Ridge coefficients as a function of the regularization', fontsize=16)
plt.legend( loc='lower left')
plt.show()