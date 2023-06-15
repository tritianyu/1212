import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings(action='ignore')

# 读入数据
data = pd.read_excel(r"C:\Users\14684\Desktop\stirpat\其他.xlsx")
dep= pd.read_excel(r"C:\Users\14684\Desktop\stirpat\碳排放.xlsx")
# 定义变量
dependent = dep["其他"]
population = data['population']
affluence = data['affluence']
urbanization = data['urbanization']
IS=data['IS']
EI=data['EI']
ES=data['ES']
dependent=dependent.values.tolist()
population=population.values.tolist()
affluence=affluence.values.tolist()
urbanization=urbanization.values.tolist()
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
IS=pd.DataFrame(IS)
EI=pd.DataFrame(EI)
ES=pd.DataFrame(ES)
X = sm.add_constant(pd.concat([population, affluence, urbanization,IS,EI,ES], axis=1))
y = dependent
# 计算岭回归系数
ridge = Ridge(alpha=0.001)
ridge.fit(X, y)
coefficients = ridge.coef_
intercept=ridge.intercept_
print(coefficients)
print(intercept)
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

# 设置各项指标的发展增速区间

POP_GROWTH_INIT = 0.0165
POP_GROWTH_STEP = -0.00015
GDP_GROWTH_INIT = 0.08
GDP_GROWTH_STEP = -0.007
URBANIZATION_GROWTH_INIT = 0.02
URBANIZATION_GROWTH_STEP = -0.001
IS_init = -0.005
IS_step = 0
EI_init = -0.04
EI_step = 0
ES_init = -0.004
ES_step = 0


def predict_next_year_data(population, gdp_per_capita, urbanization_rate,IS,EI,ES):
    """根据当前的人口规模、人均生产总值和城镇化率，以及各个指标的发展增速区间，
    计算出未来一年的数据"""
    global POP_GROWTH_INIT, GDP_GROWTH_INIT, URBANIZATION_GROWTH_INIT,  IS_init, EI_init, ES_init


    pop_growth_rate = POP_GROWTH_INIT + POP_GROWTH_STEP
    gdp_growth_rate = GDP_GROWTH_INIT + GDP_GROWTH_STEP
    urbanization_growth_rate = URBANIZATION_GROWTH_INIT + URBANIZATION_GROWTH_STEP
    IS_d_rate = IS_init + IS_step
    EI_d_rate = EI_init + EI_step
    ES_d_rate = ES_init + ES_step


    # 计算未来一年的数据
    next_year_population = population * (1 + pop_growth_rate)
    next_year_gdp_per_capita = gdp_per_capita * (1 + gdp_growth_rate)
    next_year_urbanization_rate = urbanization_rate * (1 + urbanization_growth_rate)
    next_year_IS = IS * (1 + IS_d_rate)
    next_year_EI = EI * (1 + EI_d_rate)
    next_year_ES = ES * (1 + ES_d_rate)
    #更新指标
    if POP_GROWTH_INIT > 0.0185:

      POP_GROWTH_INIT += POP_GROWTH_STEP
    if GDP_GROWTH_INIT>0.05:
       GDP_GROWTH_INIT += GDP_GROWTH_STEP
    if URBANIZATION_GROWTH_INIT>0.007:
       URBANIZATION_GROWTH_INIT += URBANIZATION_GROWTH_STEP
    if IS_init>0:
      IS_init += IS_step
    if EI_init>0:
      EI_init += EI_step
    if ES_init>0:
       ES_init += ES_step

    return next_year_population, next_year_gdp_per_capita, next_year_urbanization_rate,next_year_IS,next_year_EI,next_year_ES




def predict_next_twenty_years_data(population, gdp_per_capita, urbanization_rate,IS,EI,ES):
    """循环调用predict_next_year_data函数，预测未来20年的数据"""


    # 创建保存数据的列表
    data_list = [(population, gdp_per_capita, urbanization_rate,IS,EI,ES)]

    # 循环预测未来20年的数据
    for i in range(21):
        next_year_data = predict_next_year_data(*data_list[-1])
        data_list.append(next_year_data)

    return data_list


def save_data_to_excel(data_list, filename):
    """将保存数据的列表转换为一个DataFrame对象，并将其写入到Excel文件中"""

    # 创建DataFrame对象
    df = pd.DataFrame(data_list, columns=["population", "affluence", "urbanization","IS","EI","ES"])

    # 将DataFrame保存到Excel文件中
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, index=False)
    writer.save()


if __name__ == '__main__':
    # 初始化初始数据
    initial_population = 2523.22
    initial_gdp_per_capita = 5.43
    initial_urbanization_rate = 0.52
    IS=0.35
    EI=1.36
    ES=0.68

    # 预测未来20年的数据
    data_list = predict_next_twenty_years_data(initial_population, initial_gdp_per_capita, initial_urbanization_rate,IS,EI,ES)

    # 将数据保存到Excel文件中
    save_data_to_excel(data_list, "predicted_data.xlsx")
#读入预测数据
pred=pd.read_excel(r"predicted_data.xlsx")
population = pred['population']
affluence = pred['affluence']
urbanization = pred['urbanization']
IS=pred['IS']
EI=pred['EI']
ES=pred['ES']
population=population.values.tolist()
affluence=affluence.values.tolist()
urbanization=urbanization.values.tolist()
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
for i in range(len(IS)):
    IS[i]=math.log(IS[i])
for i in range(len(EI)):
    EI[i]=math.log(EI[i])
for i in range(len(ES)):
    ES[i]=math.log(ES[i])
population=pd.DataFrame(population)
affluence=pd.DataFrame(affluence)
urbanization=pd.DataFrame(urbanization)
IS=pd.DataFrame(IS)
EI=pd.DataFrame(EI)
ES=pd.DataFrame(ES)
X = sm.add_constant(pd.concat([population, affluence, urbanization,IS,EI,ES], axis=1))

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
save_data_to_excel(y_pred, "predicted_TAN.xlsx")