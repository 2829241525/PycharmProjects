# -*- coding: utf-8 -*-

import json
import re
import os
from random import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def dataGenerate(self):
        if os.path.exists(self.config["train_data_path"]) or not os.path.exists(self.config["valid_data_path"]):
            return #已经生成过数据集，不用再次生成
        self.date = []
        with open(self.path, encoding="utf8") as f:
            df = pd.read_csv(self.path)
            for index, row in df.iterrows():
                line = json.loads(row.to_json())
                self.data.append(line)

        random.shuffle(self.data)

        split_index = int(len(self.data) * 0.8)
        train_data = self.data[:split_index]
        test_data = self.data[split_index:]

        with open(self.config["train_data_path"], 'w', encoding="utf8") as train_file:
            for item in train_data:
                train_file.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(self.config["valid_data_path"], 'w', encoding="utf8") as test_file:
            for item in test_data:
                test_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        return


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if line.startswith("0,"):
                    label = 0
                elif line.startswith("1,"):
                    label = 1
                else:
                    continue
                title = line[2:].strip()
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/valid_tag_news.json", Config)
    print(dg[1])
