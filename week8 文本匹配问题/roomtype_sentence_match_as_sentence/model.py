# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""
#下午 00:23
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # 添加注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 添加更多特征提取层
        self.feature_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.bmm(attention_weights.transpose(1, 2), lstm_out)  # [batch_size, 1, hidden_size*2]
        attended = attended.squeeze(1)  # [batch_size, hidden_size*2]
        
        # 特征提取
        features = self.feature_layers(attended)
        features = self.dropout(features)
        
        return features


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.dropout = nn.Dropout(config["dropout"])
        
        # 增加特征交互层
        self.interaction_layer = nn.Sequential(
            nn.Linear(config["hidden_size"] * 4, config["hidden_size"] * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config["hidden_size"] * 2, 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, sentence1, sentence2, labels=None):
        # 获取句子向量
        vector1 = self.sentence_encoder(sentence1)
        vector2 = self.sentence_encoder(sentence2)
        
        # 对向量进行归一化
        vector1 = torch.nn.functional.normalize(vector1, dim=1)
        vector2 = torch.nn.functional.normalize(vector2, dim=1)
        
        # 计算特征
        abs_diff = torch.abs(vector1 - vector2)
        mul = vector1 * vector2
        
        # 拼接特征并通过交互层
        combined = torch.cat([vector1, vector2, abs_diff, mul], dim=1)
        combined = self.dropout(combined)
        logits = self.interaction_layer(combined)
        
        if labels is not None:
            loss = self.criterion(logits, labels.squeeze())
            return loss
        else:
            return torch.softmax(logits, dim=1)[:, 1]  # 返回正类的概率


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    #告诉模型是否是正样本
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())