import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math

# 读入数据
data = pd.read_excel(r"C:\Users\14684\Desktop\stirpat\交通.xlsx")
dep= pd.read_excel(r"C:\Users\14684\Desktop\stirpat\碳排放.xlsx")
pred=pd.read_excel(r"C:\Users\14684\Desktop\stirpat\pred交通.xlsx")
# 定义变量
dependent = dep["交通"]
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
ridge = Ridge(alpha=0.01)
ridge.fit(X, y)
coefficients = ridge.coef_
print(coefficients)
# 计算模型拟合优度
y_pred = ridge.predict(X)
r2 = r2_score(y, y_pred)
print(r2)
# 可视化展示
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y, color='blue', label='Original Data')
ax.plot(y_pred, color='red', label='Predicted Data')
ax.legend(loc='upper left')
plt.show()
#预测未来20年
population = pred['population']
affluence = pred['affluence']
urbanization = pred['urbanization']
m=pred['m']
n=pred['n']
IS=pred['IS']
EI=pred['EI']
ES=pred['ES']
population=population.values.tolist()
affluence=affluence.values.tolist()
urbanization=urbanization.values.tolist()
m=m.values.tolist()
n=n.values.tolist()
IS=IS.values.tolist()
EI=EI.values.tolist()
ES=ES.values.tolist()


#取对数
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
population=pd.DataFrame(population)
affluence=pd.DataFrame(affluence)
urbanization=pd.DataFrame(urbanization)
m=pd.DataFrame(m)
n=pd.DataFrame(n)
IS=pd.DataFrame(IS)
EI=pd.DataFrame(EI)
ES=pd.DataFrame(ES)
X = sm.add_constant(pd.concat([population, affluence, urbanization,m,n,IS,EI,ES], axis=1))

y_pred = ridge.predict(X)
print(y_pred)
for i in range(len(y_pred)):
    y_pred[i]=math.e**y_pred[i]

def save_data_to_excel(data_list, filename):
    """将保存数据的列表转换为一个DataFrame对象，并将其写入到Excel文件中"""

    # 创建DataFrame对象
    df = pd.DataFrame(data_list, columns=["碳排放"])

    # 将DataFrame保存到Excel文件中
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, index=False)
    writer.save()
save_data_to_excel(y_pred, "predicted_data.xlsx")