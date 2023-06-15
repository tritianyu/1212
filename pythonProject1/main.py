import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings(action='ignore')

# 读入数据
data = pd.read_excel(r"C:\Users\14684\Desktop\stirpat\电.xlsx")
dep= pd.read_excel(r"C:\Users\14684\Desktop\stirpat\碳排放.xlsx")
# 定义变量
dependent = dep["发电及供热"]
population = data['population']
affluence = data['affluence']
urbanization = data['urbanization']
m=data['m']
n=data['n']
IS=data['IS']
EI=data['EI']
ES=data['ES']
dependent=dependent.values.tolist()
population=population.values.tolist()
affluence=affluence.values.tolist()
urbanization=urbanization.values.tolist()
m=m.values.tolist()
n=n.values.tolist()
IS=IS.values.tolist()
EI=EI.values.tolist()
ES=ES.values.tolist()
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
for i in range(len(IS)):
    IS[i]=math.log(IS[i])
for i in range(len(EI)):
    EI[i]=math.log(EI[i])
for i in range(len(ES)):
    ES[i]=math.log(ES[i])
# 定义模型
dependent=pd.DataFrame(dependent)
population=pd.DataFrame(population)
affluence=pd.DataFrame(affluence)
urbanization=pd.DataFrame(urbanization)
m=pd.DataFrame(m)
n=pd.DataFrame(n)
IS=pd.DataFrame(IS)
EI=pd.DataFrame(EI)
ES=pd.DataFrame(ES)
X = sm.add_constant(pd.concat([population, affluence, urbanization,m,n,IS,EI,ES], axis=1))
y = dependent
# 计算岭回归系数
alphas = np.logspace(-3, 3, 100)
ridge = RidgeCV(alphas=alphas,cv=5)
ridge.fit(X, y)
mse = ridge.cv_values_.mean(axis=0)
# 计算模型拟合优度
# 绘制岭迹图
plt.figure(figsize=(10, 6))
plt.plot(alphas, mse, '-o', markersize=5)
plt.xlabel('Alpha', fontsize=12)
plt.ylabel('Mean Square Error', fontsize=12)
plt.title('Ridge Trace', fontsize=16)
plt.xscale('log')
plt.show()