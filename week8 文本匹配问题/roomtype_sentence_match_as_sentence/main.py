# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config):
    # 设置随机种子
    set_seed(42)
    
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.makedirs(config["model_path"], exist_ok=True)
    
    # 加载训练数据和验证数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = SiameseNetwork(config)
    
    # 使用GPU如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 训练循环
    best_accuracy = 0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"Epoch {epoch} begin")
        train_losses = []
        #训练数据量为
        train_data_size = len(train_data)
        logger.info(f"Train data size: {train_data_size}")

        for batch_idx, batch_data in enumerate(train_data):
            # 将数据移到GPU
            batch_data = [d.to(device) for d in batch_data]
            input_id1, input_id2, labels = batch_data
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            loss = model(input_id1, input_id2, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_data)}, Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = np.mean(train_losses)
        logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        # 评估模型
        evaluator.eval(epoch)
        
    # 保存最佳模型
    model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main(Config)