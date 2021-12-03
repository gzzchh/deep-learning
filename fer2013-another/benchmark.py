"""
测试各种衰减函数与模型
"""
import torch
from matplotlib import pyplot as plt
from torch import optim

import FaceModel
import train as mytrain
import FaceDataSet

trainDataSet = FaceDataSet.FaceDataSet(root="datasets-labeled", labelFile="train.csv")
testDataSet = FaceDataSet.FaceDataSet(root='datasets-labeled', labelFile='test.csv')
net = FaceModel.FaceModel()
learningRate = 0.001
epochs = 100
optimizer = optim.AdamW(net.parameters(), lr=learningRate, weight_decay=0.1)


# 根据参数动态调整
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=1, patience=3)
# 余弦退火
# 多间隔


def trainBenchmark(modelNamePrefix, optimizer, scheduler, net):
    for i in range(5):
        optimizer.zero_grad()
        model, lossRateList, learnRateList, trainAccList, testAccList, epochList = mytrain.train(trainDataSet,
                                                                                                 testDataSet, net,
                                                                                                 scheduler, optimizer,
                                                                                                 batchSize=64,
                                                                                                 epochs=epochs,
                                                                                                 learningRate=learningRate)
        # 把各种参数塞进去画图
        plt.plot(epochList, lossRateList, color="blue", label="轮数")
        plt.plot(epochList, learnRateList, color="orange", label="学习率")
        plt.plot(epochList, trainAccList, color="green", label="训练正确率")
        plt.plot(epochList, testAccList, color="red", label="测试正确率")
        plt.savefig(f"{modelNamePrefix}-{i}.png")
        torch.save(model, f"{modelNamePrefix}-{i}.pt")


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
trainBenchmark("steplr", optimizer, scheduler, net)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
trainBenchmark("cosann", optimizer, scheduler, net)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
    # 第一次调整基本上在第五次之后第十次之内
    10,
    20,
    50,
], gamma=0.1)
trainBenchmark("cosann", optimizer, scheduler, net)
