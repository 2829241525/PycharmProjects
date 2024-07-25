#coding:utf8
import torch
import torch.nn as nn

'''
embedding层的处理，根据不同类型数，给出对应级别向量
'''

num_embeddings = 7  #通常对于nlp任务，此参数为字符集字符总数,如vocab为7个字符
embedding_dim = 5   #每个字符向量化后的向量维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
print("随机初始化权重")
print(embedding_layer.weight)
print("################")

#构造字符表
vocab = {
    "a" : 0,
    "b" : 1,
    "c" : 2,
    "d" : 3,
    "e" : 4,
    "f" : 5,
    "g" : 6,
}

def str_to_sequence(string, vocab):
    return [vocab[s] for s in string]

string1 = "abcde"
string2 = "ddccb"
string3 = "fedag"

sequence1 = str_to_sequence(string1, vocab)
sequence2 = str_to_sequence(string2, vocab)
sequence3 = str_to_sequence(string3, vocab)

x = torch.LongTensor([sequence1, sequence2, sequence3])
print(x)
embedding_out = embedding_layer(x)
print(embedding_out)

