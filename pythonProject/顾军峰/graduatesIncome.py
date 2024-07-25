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
    return (y_pred - y_true)**2


#本科
w1, w2, w3 = 3.4139, 134.2464, 2370.3596
lr = 0.000001
#数据集
X=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0]
Y=[2370.0,2475.7039,2575.6484,2694.708,2799.7573,2923.7144,3033.736,3140.487,3282.8752,3389.808,3550.1025,3706.2493,3856.9736,4029.4998,4211.8916,4405.228,4566.0996,4762.18,5168.0947,5381.6177,5484.6143,5535.1196,5686.3564,5877.292]

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

X3=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0]
Y3=[2370.0,2475.7039,2575.6484,2694.708,2799.7573,2923.7144,3033.736,3140.487,3282.8752,3389.808,3550.1025,3706.2493,3856.9736,4029.4998,4211.8916,4405.228,4566.0996,4762.18,5168.0947,5381.6177,5484.6143,5535.1196,5686.3564,5877.292]
#plt.plot(X3, Y3, color="red", label='本科起薪函数')  # 原始数据

plt.scatter(X, Y, color="red" , label='本科起薪数据集图像展示')

plt.legend()

plt.xticks(ticks=plt.xticks()[0], labels=[int(i) + 2000 for i in plt.xticks()[0]])
plt.title('本科毕业起薪预测')
plt.xlabel('年份')
plt.ylabel('起薪（元）')
plt.show()
