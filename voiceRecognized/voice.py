import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MFCC
from torchaudio.models import DeepSpeech

# 定义语音识别模型训练的完整方案
def train_speech_recognition_model():
    # 加载数据集
    dataset = SPEECHCOMMANDS(root="./data", download=True, subset="training")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # 数据加载器
    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 声学特征提取：使用MFCC
    mfcc_transform = MFCC(sample_rate=16000, n_mfcc=40, log_mels=True)

    # 初始化DeepSpeech模型
    model = DeepSpeech(num_classes=35)  # 35类假设为输出类别数，例如常见的35个短语

    # 定义损失函数和优化器
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (waveform, sample_rate, label, *_ ) in enumerate(train_loader):
            # 提取MFCC特征
            inputs = mfcc_transform(waveform)
            input_lengths = torch.full((inputs.size(0),), inputs.size(2), dtype=torch.int32)
            target_lengths = torch.full((label.size(0),), len(label[0]), dtype=torch.int32)

            # 前向传播
            outputs = model(inputs)

            # 计算CTC损失
            loss = criterion(outputs.transpose(0, 1), label, input_lengths, target_lengths)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("训练完成！")

if __name__ == "__main__":
    train_speech_recognition_model()
