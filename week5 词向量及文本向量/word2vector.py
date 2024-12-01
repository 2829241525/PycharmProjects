import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 假设有一个简单的语料库
corpus = [["我", "爱", "深度", "学习"], ["深度", "学习", "很", "有趣"], ["机器", "学习", "是", "人工智能", "的", "一个", "分支"]]

# 构建词汇表
vocab = set(word for sentence in corpus for word in sentence)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocab)

# 超参数
embedding_dim = 50
context_window = 2

# 数据集构建
class CBOWDataset(Dataset):
    def __init__(self, corpus, word_to_idx, context_window):
        self.data = []
        for sentence in corpus:
            for i in range(context_window, len(sentence) - context_window):
                context = sentence[i - context_window:i] + sentence[i + 1:i + context_window + 1]
                target = sentence[i]
                context_indices = [word_to_idx[w] for w in context]
                target_index = word_to_idx[target]
                self.data.append((context_indices, target_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_indices, target_index = self.data[idx]
        return torch.tensor(context_indices, dtype=torch.long), torch.tensor(target_index, dtype=torch.long)

# 模型定义
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        # context: [batch_size, context_size]
        context_embedding = self.embeddings(context)  # [batch_size, context_size, embedding_dim]
        context_vector = torch.mean(context_embedding, dim=1)  # [batch_size, embedding_dim]
        output = self.linear(context_vector)  # [batch_size, vocab_size]
        return output

# 初始化数据集和数据加载器
dataset = CBOWDataset(corpus, word_to_idx, context_window)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 初始化模型
model = CBOWModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for context, target in dataloader:
        # 前向传播
        output = model(context)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

print("训练完成！")

# 获取词向量
word_embeddings = model.embeddings.weight.data


# 假设 word_embeddings 是训练得到的嵌入层权重，形状为 [vocab_size, embedding_dim]
# word_to_idx 是从单词到索引的映射字典

# 输出每个词的词向量
for word, idx in word_to_idx.items():
    word_vector = word_embeddings[idx]  # 获取词向量
    print(f"Word: {word}, Vector: {word_vector.numpy()}")  # 输出词和对应的向量
