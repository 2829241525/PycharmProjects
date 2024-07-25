# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x):
        return self.w1 * x ** 2 + self.w2 * x + self.w3


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset():
    base_year = 0
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




def main():
    # 配置参数
    epoch_num = 10000  # 训练轮数
    learning_rate = 0.0005  # 学习率
    # 建立模型
    model = TorchModel()
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset()
    criterion = nn.MSELoss()

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        Y_pred = model(train_x)  # 计算loss
        loss = criterion(Y_pred, train_y)  # 计算损失

        loss.backward()  # 计算梯度
        optim.step()  # 更新权重
        optim.zero_grad()  # 梯度归零
        watch_loss.append(loss.item())

        if epoch%1000 == 0:
            print(f"当前权重:w1:{torch.FloatTensor(model.w1)} w2:{torch.FloatTensor(model.w2)} w3:{torch.FloatTensor(model.w3)}")
            print(f"Epoch {epoch}: Loss {loss.item()}")

        if loss.item() < 0.001:
            print(f"当前权重:w1:{model.w1} w2:{model.w2} w3:{model.w3}")
            break

    # 预测
    Y_pred = model(train_x).detach()

    # 绘图
    plt.scatter(train_x, train_y, color="red")  # 原始数据
    plt.scatter(train_x, Y_pred, color="black")  # 预测结果
    plt.show()
    return



# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()

