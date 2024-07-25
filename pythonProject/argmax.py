import torch

# 一维张量
a = torch.tensor([1, 3, 2])
max_index = torch.argmax(a)
print(max_index)  # 输出为 1，因为 3 是最大值并且在索引 1 处

# 二维张量
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
max_indices = torch.argmax(b, dim=1)
print(max_indices)  #