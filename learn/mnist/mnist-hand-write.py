import torch
import torchvision
from torchvision import transforms, datasets
from matplotlib import pyplot as plt

# 准备转换层
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
trainset = datasets.MNIST(root='./data', train=True,
                          download=True, transform=transform)

images, labels = next(iter(trainset))
img = torchvision.utils.make_grid(images)
