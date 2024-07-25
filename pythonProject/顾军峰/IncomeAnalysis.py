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
Y=[3727.0,3853.2334,4036.2964,4212.4224,4394.862,4551.179,4751.6064,4954.478,5194.36,5435.9463,5686.702,5888.974,
   6159.246,6464.6914,6776.8975,7086.189,7394.828,7631.133,7882.245,8052.388,8191.892,8237.869,8251.599,8413.404]

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

plt.plot(X, growth_rates, color="red" , label='硕士毕业起薪预测增长趋势曲线')  # 原始数据
plt.xticks(ticks=plt.xticks()[0], labels=[int(i) + 2000 for i in plt.xticks()[0]])
plt.title('硕士毕业起薪预测增长趋势')
plt.xlabel('国内年份')
plt.ylabel('硕士毕业起薪预测增长率（%）')
pyplot.show()


