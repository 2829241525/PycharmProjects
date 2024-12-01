import numpy as np
import torch
from matplotlib import pyplot

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


# 定义模型
class PolynomialRegressionModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialRegressionModel, self).__init__()
        self.poly = PolynomialFeatures(degree)
        self.linear = nn.Linear(degree + 1, 1)  # 输入维度为多项式次数+1

    def forward(self, x):
        x_poly = self.poly.fit_transform(x)
        return self.linear(torch.FloatTensor(x_poly))


# 数据集
def build_dataset():
    base_year = 2000
    end_year = 2023
    base_salary = 2363
    years = np.arange(base_year, end_year + 1)
    growth_rates = np.random.uniform(0.04, 0.06, len(years) - 1)
    salaries = [base_salary]
    for growth_rate in growth_rates:
        salaries.append(salaries[-1] * (1 + growth_rate))
    return torch.FloatTensor(years - base_year).view(-1, 1), torch.FloatTensor(salaries).view(-1, 1)  # 归一化年份，以及调整shape


X, Y = build_dataset()

# 实例化模型，假设我们使用2次多项式
degree = 2
model = PolynomialRegressionModel(degree)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 特征转换
poly_features = PolynomialFeatures(degree)
X_poly = poly_features.fit_transform(X)

# 训练模型
num_epochs = 10000
clip_value = 10  # 梯度裁剪值
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, Y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()

    # 应用梯度裁剪来防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# 最后打印模型参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Y_pred = init_func(X, w1, w2).detach()
#
# pyplot.scatter(X.numpy(), Y.numpy(), color="red")
# pyplot.plot(X.numpy(), Y_pred.numpy(), color="black")
# pyplot.show()


# 数据预测
with torch.no_grad():  # 测试时不需要计算梯度
    predicted = model(X).data.numpy()

# 原始数据绘制
plt.scatter(X.numpy(), Y.numpy(), color='red', label='Original data')

# 模型预测绘制
plt.plot(X.numpy(), predicted, color='blue', label='Fitted line')

# 图例
plt.legend()

# 标题和标签
plt.xlabel('Year')
plt.ylabel('Salary')
plt.title('Salary Prediction')

# 显示图表
plt.show()
