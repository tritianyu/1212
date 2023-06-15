import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, brier_score_loss
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
data=pd.read_excel(r"C:\Users\14684\PycharmProjects\pythonProject3\电力.xlsx")
plt.scatter(np.arange(1998,2022),data['电力（万千瓦时）'],c='b')
plt.show()
