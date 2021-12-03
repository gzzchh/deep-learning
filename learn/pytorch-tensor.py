import torch
import numpy as np

a = torch.Tensor([1, 2, 3])
b = torch.Tensor(np.arange(1, 7).reshape(2, 3))
c = torch.Tensor(3, 2)
print(a)
print(b)
print(c)
