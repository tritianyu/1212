import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# data = pd.read_csv('jinan_6.csv',encoding='GBK')
# elec = data[['年份','年份','电力（万千瓦时）']]
# industry = set(data['行业'].tolist())
#2011nian
A = [14.8,10.5,14.6,12.6,13.3,14.8,15.2,13,12.1,12.2,12.9,15.3]
x2011 = 262685
# x2012 = 2.15696
ratio = x2011/sum(A)
B = [ratio*A[i] for i in range(len(A))]
xticks = ['2011年'+str(i)+'月' for i in range(1,13)]

plt.figure()
plt.plot(xticks,A,label='全行业用电量')
plt.bar(xticks,B,label='电力、热力的生产和供应业用电量')
plt.xlabel('时间')
plt.ylabel('行业耗电量（亿兆瓦）')
plt.title('优化插值法')
plt.legend()
plt.show()
print(B)