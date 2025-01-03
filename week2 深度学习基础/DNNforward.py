#coding:utf8

import torch
import torch.nn as nn
import numpy as np

"""
numpy手动实现模拟一个线性层
"""
#PM 00:14:30
#搭建一个2层的神经网络模型
#每层都是线性层
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        #初始化继承的模型
        super(TorchModel, self).__init__()
        #输入纬度、映射多少纬
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        #self.act = nn.GELU()

    #进行两次映射
    def forward(self, x):
        hidden = self.layer1(x)   #shape: (batch_size, input_size) -> (batch_size, hidden_size1)
        y_pred = self.layer2(hidden) #shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2)
        #y_pred = self.act(y_pred)
        return y_pred

#自定义模型
class DiyModel:
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self, x):
        hidden = np.dot(x, self.w1.T) + self.b1 #1*5
        y_pred = np.dot(hidden, self.w2.T) + self.b2 #1*2
        return y_pred



#随便准备一个网络输入
x = np.array([
                [3.1, 1.3, 1.2],
                [2.1, 1.3, 13],
                [2.1, 1.3, 13],
              ])

#建立torch模型
torch_model = TorchModel(3, 5, 2)

print(torch_model.state_dict())
print("-----------")
#打印模型权重，权重为随机初始化
torch_model_w1 = torch_model.state_dict()["layer1.weight"].numpy()
torch_model_b1 = torch_model.state_dict()["layer1.bias"].numpy()
torch_model_w2 = torch_model.state_dict()["layer2.weight"].numpy()
torch_model_b2 = torch_model.state_dict()["layer2.bias"].numpy()

#以y = wx+b 进行理解，b为偏置
print( "torch w1 权重：")
print(torch_model_w1)
print( "torch b1 权重：")
print(torch_model_b1)
print("-----------")
print("torch w2 权重：")
print(torch_model_w2)
print("torch b2 权重：")
print(torch_model_b2)
print("-----------")

#使用torch模型做预测
torch_x = torch.FloatTensor(x)
y_pred = torch_model.forward(torch_x)
print("torch模型预测结果：\n", y_pred)
# #把torch模型权重拿过来自己实现计算过程
diy_model = DiyModel(torch_model_w1, torch_model_b1, torch_model_w2, torch_model_b2)
#用自己的模型来预测
y_pred_diy = diy_model.forward(np.array(x))
print("diy模型预测结果：\n", y_pred_diy)
