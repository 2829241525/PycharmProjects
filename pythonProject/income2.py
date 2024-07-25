import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
def build_dataset():
    base_year = 2000
    end_year = 2023
    base_salary = 2363  # 假设2000年的起薪为2000元

    # 生成年份序列
    years = np.arange(base_year, end_year + 1)

    # 模拟起薪增长，假设每年增长率介于5%至10%之间，反映经济增长和通货膨胀的影响
    growth_rates = np.random.uniform(0.04, 0.06, len(years) - 1)
    salaries = [base_salary]
    for growth_rate in growth_rates:
        salaries.append(salaries[-1] * (1 + growth_rate))

    year_data = []
    for data in years:
        year_data.append(data)
    return torch.FloatTensor(year_data), torch.FloatTensor(salaries)

X, Y = build_dataset()

# 定义指数函数模型
class ExponentialModel(nn.Module):
    def __init__(self):
        super(ExponentialModel, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * torch.exp(self.b * x)

model = ExponentialModel()

# 损失函数
criterion = nn.MSELoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10000):
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"当前参数:a:{model.a.item()} b:{model.b.item()}")
        print(f"Epoch {epoch}: Loss {loss.item()}")

# 可视化拟合结果
Y_pred = model(X).detach()

plt.scatter(X.numpy(), Y.numpy(), color="red", label="原始数据")
plt.plot(X.numpy(), Y_pred.numpy(), color="black", label="拟合结果")
plt.legend()
plt.show()
