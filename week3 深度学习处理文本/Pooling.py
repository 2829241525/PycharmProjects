#coding:utf8
import torch
import torch.nn as nn

'''
pooling层的处理，缩减模型大小，提高计算速度
'''

#pooling操作默认对于输入张量的最后一维进行
#入参5，代表把五维池化为一维
layer = nn.AvgPool1d(4)
nn.MaxPool1d(4)
#随机生成一个维度为3x4x5的张量
#可以想象成3条,文本长度为4,向量长度为5的样本
x = torch.rand([3, 4, 5])
print(x)
print(x.shape)
#交换，应该对文本长度做池化,因为torch.nn.AvgPool1d默认对最后一维做池化
x = x.transpose(1,2)
print(x)
print(x.shape)
#经过pooling层，将一层进行平均
y = layer(x)
print(y)
print(y.shape)
#squeeze方法去掉值为1的维度
y = y.squeeze()
print(y)
print(y.shape)
