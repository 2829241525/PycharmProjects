# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "pretrain_model_path": r"C:\Users\Lenovo\PycharmProjects\pythonProject\week6 语言模型\bert-base-chinese",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 1,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "use_crf": True,
    "bert_path": r"C:\Users\Lenovo\PycharmProjects\pythonProject\week6 语言模型\bert-base-chinese"
}

