# -*- coding: utf-8 -*-
import torch
from loader import load_data
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def eval(self, epoch):
        self.logger.info(f"开始评估第{epoch}轮模型效果：")
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in self.valid_data:
                # 将数据移到GPU
                batch_data = [d.to(self.device) for d in batch_data]
                input_id1, input_id2, labels = batch_data
                
                # 获取预测概率
                probs = self.model(input_id1, input_id2)
                predictions = (probs > 0.7).long()
                
                # 收集预测结果和真实标签
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 统计正确和错误的数量
                for pred, label in zip(predictions, labels):
                    if pred == label:
                        self.stats_dict["correct"] += 1
                    else:
                        self.stats_dict["wrong"] += 1
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=1)
        
        # 输出评估结果
        self.logger.info(f"预测集合条目总量：{len(all_labels)}")
        self.logger.info(f"预测正确条目：{self.stats_dict['correct']}，预测错误条目：{self.stats_dict['wrong']}")
        self.logger.info(f"准确率：{accuracy:.4f}")
        self.logger.info(f"精确率：{precision:.4f}")
        self.logger.info(f"召回率：{recall:.4f}")
        self.logger.info(f"F1分数：{f1:.4f}")
        self.logger.info("--------------------")
