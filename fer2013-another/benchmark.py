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


def trainBenchmark(modelNamePrefix, schedulerChoice, net):
    """
    
    :param modelNamePrefix: 
    :param optimizer:  
    :param schedulerChoice: 在这里写选择的衰减函数名称
    :param net: 
    :return: 
    """
    for i in range(5):
        model, lossRateList, learnRateList, trainAccList, testAccList, epochList = mytrain.train(trainDataSet,
                                                                                                 testDataSet, net,
                                                                                                 schedulerChoice,
                                                                                                 batchSize=64,
                                                                                                 epochs=epochs,
                                                                                                 learningRate=learningRate)
        # 把各种参数塞进去画图
        plt.plot(epochList, lossRateList, color="blue", label="轮数")
        plt.plot(epochList, learnRateList, color="orange", label="学习率")
        plt.plot(epochList, trainAccList, color="green", label="训练正确率")
        plt.plot(epochList, testAccList, color="red", label="测试正确率")
        plt.savefig(f"{modelNamePrefix}-{i}.png")
        plt.show()
        torch.save(model, f"{modelNamePrefix}-{i}.pt")


trainBenchmark("steplr", "steplr", net)
trainBenchmark("cosann", "cosann", net)
trainBenchmark("multisteplr", "multisteplr", net)
