# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_tag_news.json",
    "valid_data_path": "data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 50,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 2,
    "batch_size": 64,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"C:\Users\Lenovo\PycharmProjects\pythonProject\week6 语言模型\bert-base-chinese",
    "seed": 987
}