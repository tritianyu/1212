import pandas as pd

# 设置各项指标的发展增速区间
POP_GROWTH_INIT = 0.0165
POP_GROWTH_STEP = 0.0001
GDP_GROWTH_INIT = 0.01
GDP_GROWTH_STEP = 0.001
URBANIZATION_GROWTH_INIT = 0.007
URBANIZATION_GROWTH_STEP = 0.0001
m_init = -0.007
m_step = 0.0001
n_init = 0.006
n_step = 0.0001
IS_init = -0.007
IS_step = 0.0001
EI_init = -0.007
EI_step = 0.0001
ES_init = -0.007
ES_step = 0.0001


def predict_next_year_data(population, gdp_per_capita, urbanization_rate,m_d,n_d,IS,EI,ES):
    """根据当前的人口规模、人均生产总值和城镇化率，以及各个指标的发展增速区间，
    计算出未来一年的数据"""

    pop_growth_rate = POP_GROWTH_INIT + POP_GROWTH_STEP
    gdp_growth_rate = GDP_GROWTH_INIT + GDP_GROWTH_STEP
    urbanization_growth_rate = URBANIZATION_GROWTH_INIT + URBANIZATION_GROWTH_STEP
    m_d_rate = m_init + m_step
    n_d_rate = n_init + n_step
    IS_d_rate = IS_init + IS_step
    EI_d_rate = EI_init + EI_step
    ES_d_rate = ES_init + ES_step

    # 计算未来一年的数据
    next_year_population = population * (1 + pop_growth_rate)
    next_year_gdp_per_capita = gdp_per_capita * (1 + gdp_growth_rate)
    next_year_urbanization_rate = urbanization_rate * (1 + urbanization_growth_rate)
    next_year_m = m_d * (1 + m_d_rate)
    next_year_n = n_d * (1 + n_d_rate)
    next_year_IS = IS * (1 + IS_d_rate)
    next_year_EI = EI * (1 + EI_d_rate)
    next_year_ES = ES * (1 + ES_d_rate)

    return next_year_population, next_year_gdp_per_capita, next_year_urbanization_rate,next_year_m,next_year_n,next_year_IS,next_year_EI,next_year_ES


def predict_next_twenty_years_data(population, gdp_per_capita, urbanization_rate,m_d,n_d,IS,EI,ES):
    """循环调用predict_next_year_data函数，预测未来20年的数据"""

    # 创建保存数据的列表
    data_list = [(population, gdp_per_capita, urbanization_rate,m_d,n_d,IS,EI,ES)]

    # 循环预测未来20年的数据
    for i in range(21):
        next_year_data = predict_next_year_data(*data_list[-1])
        data_list.append(next_year_data)

    return data_list


def save_data_to_excel(data_list, filename):
    """将保存数据的列表转换为一个DataFrame对象，并将其写入到Excel文件中"""

    # 创建DataFrame对象
    df = pd.DataFrame(data_list, columns=["Population", "GDP per capita", "Urbanization rate","m","n","IS","EI","ES"])

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
    IS=1
    EI=1
    ES=1

    # 预测未来20年的数据
    data_list = predict_next_twenty_years_data(initial_population, initial_gdp_per_capita, initial_urbanization_rate,m,n,IS,EI,ES)

    # 将数据保存到Excel文件中
    save_data_to_excel(data_list, "predicted_data.xlsx")