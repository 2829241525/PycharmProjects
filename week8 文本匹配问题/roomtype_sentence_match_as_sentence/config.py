# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "train_data_path": "../data/train.csv",
    "valid_data_path": "../data/test.csv",
    "vocab_path": "../chars.txt",
    "max_length": 50,
    "hidden_size": 128,
    "epoch": 30,
    "batch_size": 32,
    "epoch_data_size": 1000,     # 每轮训练中采样数量
    "positive_sample_rate": 0.5,  # 正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "dropout": 0.5,
}