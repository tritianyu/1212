import suiji
import openpyxl
import pandas as pd
import random
# 设置各项指标的发展增速区间
POP_GROWTH_RANGE = (0.0165, 0.021)
GDP_GROWTH_RANGE = (0.01 ,0.09)
URBANIZATION_GROWTH_RANGE = (0.007, 0.022)
m_range=(-0.007,-0.011)
n_range=(0.006,0.011)


def predict_next_year_data(population, gdp_per_capita, urbanization_rate,m_d,n_d):
    """根据当前的人口规模、人均生产总值和城镇化率，以及各个指标的发展增速区间，
    计算出未来一年的数据"""

    # 生成符合增速区间的随机数
    pop_growth_rate = random.uniform(*POP_GROWTH_RANGE)
    gdp_growth_rate = random.uniform(*GDP_GROWTH_RANGE)
    urbanization_growth_rate = random.uniform(*URBANIZATION_GROWTH_RANGE)
    m_d_rate=random.uniform(*m_range)
    n_d_rate=random.uniform(*n_range)

    # 计算未来一年的数据
    next_year_population = population * (1 + pop_growth_rate)
    next_year_gdp_per_capita = gdp_per_capita * (1 + gdp_growth_rate)
    next_year_urbanization_rate = urbanization_rate * (1 + urbanization_growth_rate)
    next_year_m=m_d*(1+m_d_rate)
    next_year_n = n_d * (1 + n_d_rate)

    return next_year_population, next_year_gdp_per_capita, next_year_urbanization_rate,next_year_m,next_year_n


def predict_next_twenty_years_data(population, gdp_per_capita, urbanization_rate,m_d,n_d):
    """循环调用predict_next_year_data函数，预测未来20年的数据"""

    # 创建保存数据的列表
    data_list = [(population, gdp_per_capita, urbanization_rate,m_d,n_d)]

    # 循环预测未来20年的数据
    for i in range(21):
        next_year_data = predict_next_year_data(*data_list[-1])
        data_list.append(next_year_data)

    return data_list


def save_data_to_excel(data_list, filename):
    """将保存数据的列表转换为一个DataFrame对象，并将其写入到Excel文件中"""

    # 创建DataFrame对象
    df = pd.DataFrame(data_list, columns=["Population", "GDP per capita", "Urbanization rate","m","n"])

    # 将DataFrame保存到Excel文件中
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, index=False)
    writer.save()


if __name__ == '__main__':
    # 初始化初始数据
    initial_population = 2523.22
    initial_gdp_per_capita = 5.43
    initial_urbanization_rate = 0.52
    m=0.77
    n=0.41

    # 预测未来20年的数据
    data_list = predict_next_twenty_years_data(initial_population, initial_gdp_per_capita, initial_urbanization_rate,m,n)

    # 将数据保存到Excel文件中
    save_data_to_excel(data_list, "predicted_data.xlsx")