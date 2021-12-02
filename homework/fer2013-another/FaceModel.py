import pandas as pd
import torch
import numpy as np
import cv2
import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn


class FaceModel(nn.Module):
    """
    这算是实现了基本的神经网络模型
    卷积+激活重复两次然后池化
    重复三次后送入全连接层
    """

    def __init__(self):
        """
        初始化网络结构
        """
        super(FaceModel, self).__init__()

        # 第一次卷积， 池化
        self.conv1 = nn.Sequential(
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，
            # 卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 1, 48, 48),
            # output:(bitch_size, 64, 48, 48), (48-3+2*1)/1+1 = 48
            # 卷积层
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),

            # 数据归一化处理，使得数据在Relu之前不会因为数据过大而导致网络性能不稳定
            # 做归一化让数据形成一定区间内的正态分布
            # 不做归一化会导致不同方向进入梯度下降速度差距很大
            nn.BatchNorm2d(num_features=64),  # 归一化可以避免出现梯度散漫的现象，便于激活。
            nn.RReLU(inplace=True),  # 激活函数

            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大值池化# output(bitch_size, 64, 24, 24)
        )

        # 第二次卷积， 池化
        self.conv2 = nn.Sequential(
            # input:(bitch_size, 64, 24, 24), output:(bitch_size, 128, 12, 12),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 第三次卷积， 池化
        self.conv3 = nn.Sequential(
            # input:(bitch_size, 128, 12, 12), output:(bitch_size, 256, 12, 12),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 最后一层不需要添加激活函数
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),

            nn.Linear(in_features=256, out_features=7),
        )

    def forward(self, x):
        """
        前向传播
        :param x: 
        :return: 
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 数据扁平化
        x = x.view(x.shape[0], -1)  # 输出维度，-1表示该维度自行判断
        y = self.fc(x)
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # y.to(device)
        return y
