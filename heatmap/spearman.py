import numpy as np
import pandas as pd
import scipy.stats as stats
def caculate_spearman_correlation(X,Y):
    return stats.spearmanr(X,Y)[0]
def caculate_spearman_correlation_p(X,Y):
    return stats.spearmanr(X,Y)[1]
data=pd.read_excel(r"C:\Users\14684\Desktop\新建文件夹\电力、热力的生产和供应业月份数据.xlsx")
x=data['电力（万千瓦时）']
y=data['碳排放']
print(caculate_spearman_correlation(x,y))
print(caculate_spearman_correlation_p(x,y))