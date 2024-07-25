import matplotlib.pyplot as pyplot
import math

X = [0.01 * i for i in range(100)]
Y = [ 6*x**2+3*x + 4 for x in X]
#Y = [math.exp(x**2) +math.sin(x)+ 3*x + 4 for x in X]


print(X)
print(Y)
pyplot.scatter(X, Y)
pyplot.show()

# X = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35000000000000003, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41000000000000003, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.5700000000000001, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.6900000000000001, 0.7000000000000001, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.9400000000000001, 0.9500000000000001, 0.96, 0.97, 0.98, 0.99]
# Y = [4.0, 4.0302, 4.0608, 4.0918, 4.1232, 4.155, 4.1872, 4.2198, 4.2528, 4.2862, 4.32, 4.3542, 4.3888, 4.4238, 4.4592, 4.495, 4.5312, 4.5678, 4.6048, 4.6422, 4.68, 4.7181999999999995, 4.7568, 4.7958, 4.8352, 4.875, 4.9152000000000005, 4.9558, 4.9968, 5.0382, 5.08, 5.122199999999999, 5.1648, 5.2078, 5.2512, 5.295, 5.3392, 5.3838, 5.4288, 5.4742, 5.5200000000000005, 5.5662, 5.6128, 5.6598, 5.7072, 5.755, 5.8032, 5.851800000000001, 5.9008, 5.9502, 6.0, 6.0502, 6.1008, 6.1518, 6.203200000000001, 6.255000000000001, 6.3072, 6.3598, 6.4128, 6.4662, 6.52, 6.5742, 6.6288, 6.6838, 6.7392, 6.795, 6.8512, 6.9078, 6.9648, 7.022200000000001, 7.08, 7.138199999999999, 7.1968, 7.2558, 7.3152, 7.375, 7.4352, 7.4958, 7.5568, 7.6182, 7.680000000000001, 7.7422, 7.8048, 7.867800000000001, 7.9312, 7.994999999999999, 8.0592, 8.1238, 8.1888, 8.2542, 8.32, 8.3862, 8.4528, 8.5198, 8.587200000000001, 8.655000000000001, 8.7232, 8.7918, 8.8608, 8.9302]

#[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35000000000000003, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41000000000000003, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.5700000000000001, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.6900000000000001, 0.7000000000000001, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.9400000000000001, 0.9500000000000001, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001, 1.1500000000000001, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001, 1.4000000000000001, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6, 1.61, 1.62, 1.6300000000000001, 1.6400000000000001, 1.6500000000000001, 1.6600000000000001, 1.67, 1.68, 1.69, 1.7, 1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.8800000000000001, 1.8900000000000001, 1.9000000000000001, 1.9100000000000001, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99]
#[4.0, 4.030002, 4.060016, 4.090054, 4.120128, 4.15025, 4.180432, 4.210686, 4.241024, 4.271458, 4.302, 4.332662, 4.363456, 4.394394, 4.425488, 4.4567499999999995, 4.488192, 4.519826, 4.551664, 4.583718, 4.616, 4.648522, 4.681296, 4.714334, 4.747648, 4.78125, 4.815152, 4.849366, 4.883904, 4.918778, 4.954, 4.989582, 5.025536, 5.0618739999999995, 5.0986080000000005, 5.13575, 5.173312, 5.2113059999999995, 5.249744, 5.288638, 5.328, 5.367842, 5.408176, 5.449014, 5.490368, 5.53225, 5.574672, 5.617646000000001, 5.661184, 5.705298, 5.75, 5.7953019999999995, 5.841216, 5.887754, 5.934928, 5.98275, 6.031232, 6.080386000000001, 6.130224, 6.180758, 6.231999999999999, 6.283962, 6.336656, 6.390094, 6.444288, 6.49925, 6.554992, 6.6115260000000005, 6.668864, 6.727018000000001, 6.7860000000000005, 6.845822, 6.906496000000001, 6.968033999999999, 7.030448, 7.09375, 7.157952, 7.223066, 7.289104, 7.356078, 7.424, 7.492882, 7.562736, 7.633574, 7.705408, 7.77825, 7.852112, 7.9270059999999996, 8.002944, 8.079938, 8.158000000000001, 8.237142, 8.317376, 8.398714, 8.481168, 8.56475, 8.649472, 8.735346, 8.822384, 8.910598, 9.0, 9.090602, 9.182416, 9.275454, 9.369728, 9.465250000000001, 9.562032, 9.660086, 9.759424000000001, 9.860058, 9.962000000000002, 10.065262, 10.169856000000001, 10.275794000000001, 10.383088, 10.491750000000001, 10.601792, 10.713225999999999, 10.826063999999999, 10.940318, 11.056, 11.173122, 11.291696, 11.411734, 11.533248, 11.65625, 11.780752, 11.906766000000001, 12.034304, 12.163378000000002, 12.294, 12.426182, 12.559936, 12.695274000000001, 12.832208000000001, 12.970750000000002, 13.110912, 13.252706000000002, 13.396144000000003, 13.541238000000002, 13.688000000000002, 13.836441999999998, 13.986576, 14.138414, 14.291968, 14.44725, 14.604271999999998, 14.763046, 14.923583999999998, 15.085898, 15.25, 15.415901999999999, 15.583616000000001, 15.753154, 15.924528, 16.09775, 16.272832, 16.449786, 16.628624000000002, 16.809358000000003, 16.992000000000004, 17.176562000000004, 17.363056, 17.551494, 17.741888000000003, 17.934250000000002, 18.128592000000005, 18.324925999999998, 18.523263999999998, 18.723618000000002, 18.926, 19.130422, 19.336896, 19.545434, 19.756048, 19.96875, 20.183552, 20.400466, 20.619504, 20.840678, 21.064, 21.289482, 21.517136, 21.746974, 21.979008, 22.213250000000002, 22.449712, 22.688406000000004, 22.929344, 23.172538000000003, 23.418000000000003, 23.665742, 23.915775999999997, 24.168114, 24.422767999999998, 24.67975, 24.939072, 25.200746, 25.464784, 25.731198]


def init_func(x):
    y = w1 * x ** 2 + w2 * x + w3
    return y

##损失函数
def loss(y_pred, y_true,len):
    return (y_pred - y_true)**2



# 权重随机初始化
w1, w2, w3 = 1, -1, 1

# 学习率
lr = 0.1
batch_size = 20

for epoch in range(10000):
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

        w1 = w1 - lr * grad_w1
        w2 = w2 - lr * grad_w2
        w3 = w3 - lr * grad_w3

        # count += 1
        # if count == batch_size:
        #     w1 = w1 - lr * grad_w1/batch_size
        #     w2 = w2 - lr * grad_w2/batch_size
        #     w3 = w3 - lr * grad_w3/batch_size
        #     count = 0
        #     grad_w1 = 0
        #     grad_w2 = 0
        #     grad_w3 = 0

    epoch_loss /= len(X)
    print(f"当前权重:w1:{w1} w2:{w2} w3:{w3}")
    print("第%d轮， loss %f" %(epoch, epoch_loss))
    if epoch_loss < 0.001:
        break

print(f"训练后权重:w1:{w1} w2:{w2} w3:{w3}")


Y1 = [init_func(i) for i in X]

pyplot.scatter(X, Y, color="red")
pyplot.scatter(X, Y1,color ="black")
pyplot.show()