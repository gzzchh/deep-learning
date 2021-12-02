"""
自动求导
"""
import torch
from torch import autograd

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

z = x**2 + y
grads = autograd.grad(z, [x, y])
print(grads[0], grads[1])
