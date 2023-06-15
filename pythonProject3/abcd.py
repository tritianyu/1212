import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import lagmat
from statsmodels import QuantReg

# 假设我们有一组时间序列数据 X 和 Y
# X 可以看作是自变量，Y 是因变量
data = {'X': [x1, x2, x3, ..., xn], 'Y': [y1, y2, y3, ..., yn]}
df = pd.DataFrame(data)

# 指定 X 和 Y 的滞后期数
lags_x = 2
lags_y = 2

# 生成滞后项
X_lags = lagmat(df['X'], maxlag=lags_x, trim='both')
Y_lags = lagmat(df['Y'], maxlag=lags_y, trim='both')

# 构建 ARDL 模型
# 以 ARDL(2, 2) 为例，这里的 2 表示两个滞后期
X_lags = sm.add_constant(X_lags)
Y_lags = sm.add_constant(Y_lags)
y = df['Y'][(lags_y + 1):]
X = np.column_stack([Y_lags[(lags_y + 1 - i):(len(Y_lags) - i), :] for i in range(1, lags_y + 1)] + [X_lags[(lags_x + 1 - i):(len(X_lags) - i), :] for i in range(1, lags_x + 1)] + [df['X'][(lags_y + 1):]])
model = sm.OLS(y, X).fit()

# 进行分位数回归
# 假设我们要估算在不同分位数下 Y 对 X 的反应程度
quantiles = [0.25, 0.5, 0.75]
results = []
for quantile in quantiles:
    # 通过 quantile 固定分位数，计算分位数回归系数
    qmod = QuantReg(y, X)
    qres = qmod.fit(q=quantile)
    results.append([quantile, qres.params])

# 输出结果
print(results)