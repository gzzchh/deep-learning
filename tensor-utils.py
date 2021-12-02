import torch

# 准备3维的张量
a = torch.tensor([[[5.5]]])
# 打印所有内容
print(a.item())
# 打印维度
print(a.dim())
# 3 行 2 列
b = torch.rand(3, 2)
print(b.numpy())
print(b.dim())
print(b.size())
# 压到一维显示
print(b.view(6))

c = torch.rand(2, 3)
print(c)
print(c.t())
# 2 维 3 行 4 列
d = torch.arange(0, 24).view(2, 3, 4)
print(d)
print(d.transpose(1,2))
print(d.permute(0, 2, 1))
