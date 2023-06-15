import pandas as pd
from matplotlib import pyplot as plt
import random

from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
import numpy as np

random_float = random.random()

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

A2010 = [12.6, 10.4, 14.1, 12.6, 12.9, 12.9, 14.9, 13.1, 11.7, 12.8, 13.2, 14.8]
A2011 = [14.8, 10.5, 14.6, 12.6, 13.3, 14.8, 15.2, 13, 12.1, 12.2, 12.9, 15.3]
A2012 = [13.1, 11.6, 13.1, 12.19, 13.34, 13.32, 14.2, 10.7, 10.6, 11.1, 12.26, 14]
A2013 = [8.5, 7.41, 14.7, 11.4, 12.36, 12.7, 14, 14.1, 9.03, 11.61, 13.7, 10.1]

elect_all = {'1998': [i + 2 * random.random() - 1 for i in A2010], '1999': [i + 2 * random.random() - 1 for i in A2011],
             '2000': [i + 2 * random.random() - 1 for i in A2012], '2001': [i + 2 * random.random() - 1 for i in A2013],
             '2002': [i + 2 * random.random() - 1 for i in A2010], '2003': [i + 2 * random.random() - 1 for i in A2011],
             '2004': [i + 2 * random.random() - 1 for i in A2012], '2005': [i + 2 * random.random() - 1 for i in A2013],
             '2006': [i + 2 * random.random() - 1 for i in A2010], '2007': [i + 2 * random.random() - 1 for i in A2011],
             '2008': [i + 2 * random.random() - 1 for i in A2012], '2009': [i + 2 * random.random() - 1 for i in A2013],
             '2014': [i + 2 * random.random() - 1 for i in A2013], '2015': [i + 2 * random.random() - 1 for i in A2012],
             '2016': [i + 2 * random.random() - 1 for i in A2012], '2017': [i + 2 * random.random() - 1 for i in A2010],
             '2018': [i + 2 * random.random() - 1 for i in A2011], '2019': [i + 2 * random.random() - 1 for i in A2012],
             '2020': [i + 2 * random.random() - 1 for i in A2013], '2021': [i + 2 * random.random() - 1 for i in A2010],
             '2010': [i for i in A2010], '2011': [i for i in A2011], '2012': [i for i in A2012],
             '2013': [i for i in A2013]}

data = pd.read_csv('jinan_6.csv', encoding='GBK')
industry = list(set(data['行业'].tolist()))
##能源类型
energy = data.columns[2:]
# 行业
consume = pd.DataFrame(columns=data.columns)
all_consume = pd.DataFrame(columns=data.columns)
for item in industry:
    industry_data = data[data['行业'] == item]
print(energy)