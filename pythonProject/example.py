# 导入库
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from scipy.optimize import leastsq


# 多项式函数
# np.polyld([c1,c0])函数用于生成多项式函数c1x^1+c0x^0
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


# 残差函数
def error_func(p, x, y):
    err = fit_func(p, x) - y
    return err


# 需要使用numpy.array()将样本数据(xi,yi)转换成数组形式
x_data = np.array([1.51, 1.64, 1.6, 1.73, 1.82, 1.87])  # 创建x列表存储数据的x值
y_data = np.array([1.63, 1.7, 1.71, 1.72, 1.76, 1.86])  # 创建y列表存储数据的y值
x_predict = np.arange(1.5, 2, 0.1)  # 预测区间


# 使用最小二乘函数求解参数
# leastsq()函数位于Scipy的optimize模块中，利用leastsq()函数可以对数据进行拟合。
# 参数residuals_func表示误差函数
# 参数p_init表示初始c0、c1
# 数据点
p_init = np.random.randn(2)   # 生成2个随机数，作为各项系数的初始值，用于迭代修正
plsq = leastsq(error_func, p_init, args=(x_data, y_data))
# leastsq返回的是一个元组，元组的第一个元素是多项式系数数组
print("[c0,c1] = ", end='')
print(plsq[0])


# 绘制图像
plt.scatter(x_data, y_data, label="样本点", color="black", zorder=2)
plt.plot(x_predict, fit_func(plsq[0], x_predict), label="拟合结果", color="red", zorder=1)
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.title("父亲身高(x)与孩子身高(y)数据集")
plt.legend(loc="upper left")
plt.show()


