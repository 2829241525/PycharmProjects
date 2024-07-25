import matplotlib.pyplot as pyplot
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


#本科

X=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0]
Y=[2370.0,2475.7039,2575.6484,2694.708,2799.7573,2923.7144,3033.736,3140.487,3282.8752,3389.808,3550.1025,3706.2493,3856.9736,4029.4998,4211.8916,4405.228,4566.0996,4762.18,5168.0947,5381.6177,5484.6143,5535.1196,5686.3564,5877.292]

#硕士
X2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
Y2=[3727.0,3853.2334,4036.2964,4212.4224,4394.862,4551.179,4751.6064,4954.478,5194.36,5435.9463,5686.702,5888.974,
   6159.246,6464.6914,6776.8975,7086.189,7394.828,7631.133,7882.245,8052.388,8191.892,8237.869,8251.599,8413.404]


plt.plot(X, Y, color="red", label='本科起薪函数')  # 原始数据
plt.plot(X2, Y2,color ="black",label='硕士起薪拟合函数')
plt.legend()
plt.xticks(ticks=plt.xticks()[0], labels=[int(i) + 2000 for i in plt.xticks()[0]])
plt.title('硕士毕业起薪预测')
plt.xlabel('年份')
plt.ylabel('起薪（元）')
plt.show()
