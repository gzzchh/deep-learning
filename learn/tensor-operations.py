import torch

# 在 1 到 25 中生成随机数
# 25 维 3x4 矩阵
a = torch.arange(1, 25).view(2, 3, 4)
print(a)
# 第一个矩阵 坐标 3 3 设定维 0
a[0, 2, 2] = 0
a[0, 1, 3] = 0
