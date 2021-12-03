import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepEmotionModel(nn.Module):
    def __init__(self):
        '''
        在这里定义模型的结构
        '''
        super(DeepEmotionModel, self).__init__()
        # 卷积和池化层,老经典操作了属于是
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(10, 10, 3)
        self.conv4 = nn.Conv2d(10, 10, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 归一化
        self.norm = nn.BatchNorm2d(10)

        # 定义两个全连接层 
        self.fc1 = nn.Linear(810, 50)
        self.fc2 = nn.Linear(50, 7)

        self.localization = nn.Sequential(
            # 这一部分可以说是老典型了,卷一下池化一下再激活
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            # 设定全连接层
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # 权重扔掉 加入偏置
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        # 给你拍平了 好送去全连接
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        # 此处也一样,但是注意大小
        theta = theta.view(-1, 2, 3)
        
        # 进行仿射变换 增强学习效率
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input):
        out = self.stn(input)

        # 处理完的图像,又是拿来做经典操作了 
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        # 手工丢弃一下,自然是为了防止过拟合
        out = F.dropout(out)
        # 全连接前要拉平
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
