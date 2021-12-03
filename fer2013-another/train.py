from enum import Enum

from torch.optim.lr_scheduler import ReduceLROnPlateau

import FaceModel
import FaceDataSet
import DeepEmotionModel
import pandas as pd
import torch
import numpy as np
import cv2
import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Shake(Enum):
    VANILLA = 7
    CHOCOLATE = 4
    COOKIES = 9
    MINT = 3


# 训练模型
def train(trainSet, validateSet, model, scheduler, optimizer, batchSize, epochs, learningRate):
    # 选择设备
    # 载入数据并分割batch
    train_loader = data.DataLoader(trainSet, batchSize, shuffle=True)
    # 构建模型
    # model = FaceModel.FaceModel()
    # 使用 DeepEmotion 的时候需要调整数据加载器
    # model = DeepEmotionModel.DeepEmotionModel()
    # 损失函数
    lossFunction = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=learningRate, weight_decay=wtDecay)
    # optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=wtDecay)
    # optimizer = optim.AdamW(model.parameters(), lr=learningRate, weight_decay=0.1)
    # 学习率衰减
    # 逐步调整,10 次一减
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 根据参数动态调整
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=1, patience=3)
    # 余弦退火
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
    # 多间隔
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
    #     # 第一次调整基本上在第五次之后第十次之内
    #     10,
    # ], gamma=0.1)
    lossRateList = []
    learnRateList = []
    trainAccList = []
    testAccList = []
    epochList = []
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        lossRate = 0
        # scheduler.step()
        # 注意dropout网络结构下训练和test模式下是不一样的结构
        # 你得开到训练模式
        model.train()
        model.to(device)
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model.forward(images)
            lossRate = lossFunction(output.cuda(), labels.cuda())
            # lossRate = lossFunction(output, labels)
            lossRate.backward()
            optimizer.step()
            # 损失率不再减小就缩小学习率,适用于根据参数动态调整
            # scheduler.step(lossRate)
            # 余弦退火
            scheduler.step()
        # 测试模型
        # 打印每轮的损失
        print(f"已训练 {epoch + 1} 轮,学习率是 {scheduler.get_last_lr()}")
        print(f"已训练 {epoch + 1} 轮,损失率是 {lossRate.item()}")
        lossRateList.append(lossRate.item())
        learnRateList.append(scheduler.get_last_lr())
        epochList.append(epoch)
        if (epoch + 1) % 5 == 0:
            # 开启测试模式 此时训练不会产生影响
            model.eval()
            model.to(device)
            trainAcc = validate(model, trainSet, batchSize)
            testAcc = validate(model, validateSet, batchSize)
            # 用测试集的准确率来更新学习率
            print(f"已训练 {epoch + 1} 轮 acc_train (训练正确率) 是 : {trainAcc}")
            print(f"已训练 {epoch + 1} 轮 acc_val (验证正确率) 是 : {testAcc}")
            trainAccList.append(trainAcc)
            testAccList.append(testAcc)

    return model, lossRateList, learnRateList, trainAccList, testAccList,epochList


def validate(model, dataset, batchSize):
    """
    验证模型在验证集上的正确率
    :param model: 模型
    :param dataset: 数据集
    :param batchSize: 批处理大小
    :return: 
    """
    validateLoader = data.DataLoader(dataset, batchSize, shuffle=True)
    result, num = 0.0, 0
    for images, labels in validateLoader:
        pred = model.forward(images)
        # print(pred)
        pred = np.argmax(pred.cpu().data.numpy(), axis=1)
        labels = labels.data.numpy()
        result += np.sum((pred == labels))
        num += len(images)

    acc = result / num
    return acc


trainDataSet = FaceDataSet.FaceDataSet(root="datasets-labeled", labelFile="train.csv")
testDataSet = FaceDataSet.FaceDataSet(root='datasets-labeled', labelFile='test.csv')


def doTrain(scheduler, model, modelName=""):
    # 超参数可自行指定
    model = train(trainDataSet, testDataSet, model,
                  scheduler, batchSize=64, epochs=100, learningRate=0.001, wtDecay=0.3)
    # 保存模型
    torch.save(model, modelName)
