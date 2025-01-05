# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import pandas as pd

"""
数据加载
"""

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.train_data_size = config["epoch_data_size"]
        self.data = []
        self.knwb = defaultdict(list)
        self.load()

    def load(self):
        try:
            df = pd.read_csv(self.path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(self.path, encoding='latin1')
            except:
                df = pd.read_csv(self.path, encoding='gb18030')
        
        # 确保列名正确
        if 'text1' not in df.columns or 'text2' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV文件必须包含text1, text2和label列")

        # 数据预处理
        df['text1'] = df['text1'].astype(str).apply(lambda x: x.encode('latin1').decode('gbk',errors='ignore'))
        df['text2'] = df['text2'].astype(str).apply(lambda x: x.encode('latin1').decode('gbk',errors='ignore'))
        df['label'] = df['label'].astype(int)

        # 转换文本为ID序列并存储
        for _, row in df.iterrows():
            text1_ids = self.encode_sentence(row['text1'])
            text2_ids = self.encode_sentence(row['text2'])
            label = row['label']

            # 转换为tensor
            text1_tensor = torch.LongTensor(text1_ids)
            text2_tensor = torch.LongTensor(text2_ids)
            label_tensor = torch.LongTensor([label])

            if "train" in str(self.path).lower():
                self.knwb[label].append((text1_tensor, text2_tensor))
            else:
                self.data.append((text1_tensor, text2_tensor, label_tensor))

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if "train" in str(self.path).lower():
            return self.train_data_size
        else:
            return len(self.data)

    def __getitem__(self, index):
        if "train" in str(self.path).lower():
            return self.random_train_sample()
        else:
            return self.data[index]

    def random_train_sample(self):
        # 随机选择标签
        if random.random() < self.config["positive_sample_rate"]:
            # 生成正样本
            label = 1
            if len(self.knwb[label]) > 0:
                text1, text2 = random.choice(self.knwb[label])
            else:
                # 如果没有正样本，则从负样本中选择
                text1, text2 = random.choice(self.knwb[0])
                label = 0
        else:
            # 生成负样本
            label = 0
            if len(self.knwb[label]) > 0:
                text1, text2 = random.choice(self.knwb[label])
            else:
                # 如果没有负样本，则从正样本中选择
                text1, text2 = random.choice(self.knwb[1])
                label = 1
        
        return text1, text2, torch.LongTensor([label])

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/train.csv", Config)
    print(dg[1])
