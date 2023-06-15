import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'http://www.spiderbuf.cn/beginner?level=1'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 从BeautifulSoup对象中找到表格内容
table = soup.find('table', class_='table').find('tbody')

# 遍历表格内容并保存到列表中
data = []
for row in table.find_all('tr'):
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

# 将列表转化为DataFrame对象
df = pd.DataFrame(data)

# 将DataFrame对象保存到本地CSV文件中
df.to_csv('table.csv', index=False)

