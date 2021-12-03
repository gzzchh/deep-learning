import os

import pandas as pd
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms


class FaceDataSet(data.Dataset):
    """
    这不得自己整个数据集?
    """

    # 初始化
    def __init__(self, root, labelFile):
        """
        传入数据集的路径
        :param root: 传入路径,写到 dataset-labeled 即可
        :param labelFile: 写文件名 只要文件名
        """
        super(FaceDataSet, self).__init__()
        self.root = root
        # 首先是数据集路径 然后是标签文件
        df = pd.read_csv(os.path.join(root, labelFile), header=None)
        # 先标记后路径
        dfLabel = df[[df.columns[0]]]
        dfPath = df[[df.columns[1]]]
        # 再给他整成 numpy
        self.path = np.array(dfPath)[:, 0]
        self.label = np.array(dfLabel)[:, 0]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def __getitem__(self, item):
        """
        通过索引获取数据集中的样本
        :param item: 索引
        :return: 样本
        """
        face = cv2.imread(os.path.join(self.root, self.path[item]))  # 读取图片
        # BGR转灰度
        faceInGray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 直方图均衡化
        # faceHist = cv2.equalizeHist(faceInGray)
        # 像素值标准化,0-255的像素范围转成0-1范围来描述
        # faceNormalized = faceHist.reshape(1, 48, 48) / 255.0
        faceNormalized = faceInGray.reshape(1, 48, 48) / 255.0
        # 适配到 torch 的类型
        # faceTensor = torch.from_numpy(faceNormalized)
        # 给 DeepEmotion 用
        faceTensor = self.transform(Image.open(os.path.join(self.root, self.path[item])))
        if torch.cuda.is_available():
            # 此时转换为 CUDA 版本
            faceTensor = faceTensor.type("torch.cuda.FloatTensor")
        else:
            faceTensor = faceTensor.type("torch.FloatTensor")
        label = self.label[item]
        # print(label)
        return faceTensor, label

    def __len__(self):
        """
        返回数据集中样本的数量
        :return: 返回的数量
        """
        return self.path.shape[0]
