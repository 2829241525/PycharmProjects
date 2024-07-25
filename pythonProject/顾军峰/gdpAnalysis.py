import matplotlib.pyplot as pyplot
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

#数据集
X=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0]
Y = [10.03,11.09,12.17,13.74,16.18,18.73,21.94,27.00,31.92,34.85,41.21,48.79,53.86,59.3,64.36,68.89,74.64,83.20,91.93,98.65,101.36,114.92,120.47,126.06]

plt.scatter(X, Y, color="red" , label='GDP')  # 原始数据
plt.xticks(ticks=plt.xticks()[0], labels=[int(i) + 2000 for i in plt.xticks()[0]])
plt.title('国内生产总值历年数据')
plt.xlabel('国内年份')
plt.ylabel('国内生产总值（万亿）')
pyplot.show()



# 初始化增长率列表
growth_rates = []
growth_rates.append(0)
# 计算每个数据点的增长率
for i in range(1, len(Y)):
    growth_rate = ((Y[i] - Y[i - 1]) / Y[i - 1]) * 100  # 增长率计算公式
    growth_rates.append(growth_rate)

# 打印增长率
for i, rate in enumerate(growth_rates, start=1):
    print(f"第{i}年的增长率: {rate:.2f}%")

plt.scatter(X, growth_rates, color="red" , label='GDP增长率波动曲线')  # 原始数据
plt.xticks(ticks=plt.xticks()[0], labels=[int(i) + 2000 for i in plt.xticks()[0]])
plt.title('GDP增长率')
plt.xlabel('国内年份')
plt.ylabel('GDP增长率百分比（%）')
pyplot.show()


