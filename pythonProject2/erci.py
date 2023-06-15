import pandas as pd
A = [14.8,10.5,14.6,12.6,13.3,14.8,15.2,13,12.1,12.2,12.9,15.3]
x1 =[50159,
37628,
72937,
84546,
106669,
137925,
96582,
34391,
40362,
11520,
13675,
9704,
7933,
5322,
3110,
2866.94,
4426,
1380,
0,
128,
0,
75382,
20842,
0
]
x2=[435,
389,
441,
2356,
213,
1276,
678,
206,
362,
102,
355,
462,
456,
411,
147,
127.75,
135.95,
95,
91,
33,
19,
14,
16,
27

]
x3=[89,
78,
73,
345,
10,
7,
7,
2,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0

]
x4=[668,
791,
263,
36,
417,
544,
634,
66,
29,
96,
391,
198,
252,
285,
35,
23.43,
2.4,
2,
9,
2,
13,
6,
11,
18


]
x5=[483,
521,
545,
651,
781,
960,
0,
428,
345,
283,
0,
210,
179,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0


]
x6=	[312659,
330618,
383228,
174286,
172853,
258799,
210111,
145512,
20000,
16000,
18947,
18473,
0,
7949,
9195,
8555,
0,
0,
0,
0,
213201,
269930,
318891,
1089481

]
x7=	[4758,
11845,
15944,
8569,
6521,
11219,
8757,
6989,
7358,
6182,
2964,
2908,
4066,
3937,
3490,
5497.26,
5930.63,
6778,
6172,
6173,
7131,
21644,
16538,
18102


]
ratio = [A[i]/sum(A)for i in range(len(A))]
B = [ratio[i]*x1[j] for j in range(0,24) for i in range(len(ratio)) ]#原煤
C=[ratio[i]*x2[j] for j in range(0,24) for i in range(len(ratio))]#汽油
D=[ratio[i]*x3[j] for j in range(0,24) for i in range(len(ratio))]#煤油
E=[ratio[i]*x4[j] for j in range(0,24) for i in range(len(ratio))]#柴油
F=[ratio[i]*x5[j] for j in range(0,24) for i in range(len(ratio))]#燃料油
G=[ratio[i]*x6[j] for j in range(0,24) for i in range(len(ratio))]#热力
H=[ratio[i]*x7[j] for j in range(0,24) for i in range(len(ratio))]#电力（亿千瓦时）
orderid=a=pd.date_range('1998-01','2021-12',freq='MS').strftime("%Y-%m")
testdata=[orderid,B,C,D,E,F,G,H]
filename2='测试.xlsx'
def pd_toexcel(data,filename):
    dfData={'年月':data[0],
    '原煤':data[1],
    '汽油':data[2],
    '煤油':data[3],
    '柴油':data[4],
    '燃料油':data[5],
    '热力':data[6],
    '电力（万千瓦时）':data[7],}
    df=pd.DataFrame(dfData)
    df.to_excel(filename,index=False)
pd_toexcel(testdata,filename2)

