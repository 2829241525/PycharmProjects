import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 创建数据集
# X = torch.tensor([0.01 * i for i in range(100)], dtype=torch.float32)
# Y = torch.tensor([6 * x ** 2 + 3 * x + 4 for x in X], dtype=torch.float32)
# def build_dataset():
#     base_year = 2000
#     end_year = 2023
#     base_salary = 2363  # 假设2000年的起薪为2000元
#
#     # 生成年份序列
#     years = np.arange(base_year, end_year + 1)
#
#     # 模拟起薪增长，假设每年增长率介于5%至10%之间，反映经济增长和通货膨胀的影响
#     growth_rates = np.random.uniform(0.04, 0.06, len(years) - 1)
#     salaries = [base_salary]
#     for growth_rate in growth_rates:
#         salaries.append(salaries[-1] * (1 + growth_rate))
#
#     year_data = []
#     for data in years:
#         year_data.append(data)
#     return torch.FloatTensor(year_data), torch.FloatTensor(salaries)
# X,Y = build_dataset()

X = torch.tensor([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], dtype=torch.float32)
Y = torch.tensor([2363, 2463.163133724215, 2561.808910659491, 2706.2288744043576, 2816.22754929814, 2960.817832542436, 3094.321720244651, 3237.1858077661173, 3377.271993737033, 3516.5812563992777, 3675.5232631900426, 3825.506041000528, 3988.074532337531, 4151.6743136894165, 4339.058535242235, 4528.438912265908, 4721.912265640934, 4926.908154911771, 5141.525263671098, 5417.02292507195, 5711.372659149421, 6052.4238312267225, 6346.688998417895, 6711.680351153348], dtype=torch.float32)


# 定义模型
class PolynomialModel(nn.Module):
    def __init__(self):
        super(PolynomialModel, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.w1 * x ** 2 + self.w2 * x + self.w3


model = PolynomialModel()

# 损失函数
criterion = nn.MSELoss()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
for epoch in range(10000):
    optimizer.zero_grad()  # 清空梯度
    Y_pred = model(X)  # 前向传播
    loss = criterion(Y_pred, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    if epoch % 100 == 0:
        print(f"当前权重:w1:{model.w1} w2:{model.w2} w3:{model.w3}")
        print(f"Epoch {epoch}: Loss {loss.item()}")

    if loss.item() < 0.001:
        print(f"当前权重:w1:{model.w1} w2:{model.w2} w3:{model.w3}")
        break

# 预测
Y_pred = model(X).detach()

# 绘图
plt.scatter(X, Y, color="red")  # 原始数据
plt.scatter(X, Y_pred, color="black")  # 预测结果
plt.show()
