#coding:utf8
import torch
import numpy

'''
softmax的计算
'''

def softmax(x):
    res = []
    for i in x:
        # e的1次方
        res.append(numpy.exp(i))
    res = [r / sum(res) for r in res]
    return res


#x = [0.1,2,3]
x = [0.3,0.1,0.3]
#torch实现的softmax
print(torch.softmax(torch.Tensor(x), 0))
#自己实现的softmax
print(softmax(x))