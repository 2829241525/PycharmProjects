import matplotlib.pyplot as pyplot
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


def init_func(x):
    y = w1 * x ** 2 + w2 * x + w3
    return y

##损失函数
def loss(y_pred, y_true,len):
    return (y_pred - y_true)**2/len


#硕士


w1, w2, w3 = 1,1,3727






w1, w2, w3 = 1.51, 193, 3623.23
lr = 0.000005 #学习率
#数据集
X=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
Y=[3727.0,3853.2334,4036.2964,4212.4224,4394.862,4551.179,4751.6064,4954.478,5194.36,5435.9463,5686.702,5888.974,
   6159.246,6464.6914,6776.8975,7086.189,7394.828,7631.133,7882.245,8052.388,8191.892,8237.869,8251.599,8413.404]

plt.scatter(X, Y, color="red")  # 原始数据
pyplot.show()


batch_size = 20


plt.scatter(X, Y, color="red")  # 原始数据
pyplot.show()

for epoch in range(10001):
    epoch_loss = 0
    count = 0
    grad_w1 = 0
    grad_w2 = 0
    grad_w3 = 0
    for x, y_true in zip(X, Y):
        #预测值
        y_pred = init_func(x)
        #计算损失
        epoch_loss += loss(y_pred, y_true, len(X))
        #梯度计算,分别对w1，w2，w3进行求导，求导函数是（w1 * x**2 + w2 * x + w3-y_true）**2
        grad_w1 = 2 * (y_pred - y_true) * x ** 2
        grad_w2 = 2 * (y_pred - y_true) * x
        grad_w3 = 2 * (y_pred - y_true)
        #优化器
        w1 = w1 - lr * grad_w1
        w2 = w2 - lr * grad_w2
        w3 = w3 - lr * grad_w3


    epoch_loss /= len(X)
    if epoch%1000 == 0:
        print(f"当前权重:w1:{w1} w2:{w2} w3:{w3}")
        print("第%d轮， loss %f" %(epoch, epoch_loss))
    if epoch_loss < 0.001:
        break

print(f"训练后权重:w1:{w1} w2:{w2} w3:{w3}")

X2=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0]
Y1 = [init_func(i) for i in X2]
print(Y1)

plt.scatter(X, Y, color="red" , label='硕士起薪数据集图像展示')
#plt.plot(X2, Y1,color ="black",label='硕士起薪拟合函数')
plt.scatter(X2, Y1,color ="black" ,label='硕士起薪函数预测图像展示')

plt.legend()

plt.xticks(ticks=plt.xticks()[0], labels=[int(i) + 2000 for i in plt.xticks()[0]])
#plt.title('硕士与本科历年毕业起薪趋势')
plt.title('硕士毕业起薪预测')
plt.xlabel('年份')
plt.ylabel('起薪（元）')
plt.show()
