"""
测试各种衰减函数与模型
"""
import torch
from matplotlib import pyplot as plt
from torch import optim
import DeepEmotionModel
import FaceModel
import train as mytrain
import FaceDataSet

trainDataSet = FaceDataSet.FaceDataSet(root="datasets-labeled", labelFile="train.csv")
testDataSet = FaceDataSet.FaceDataSet(root='datasets-labeled', labelFile='test.csv')
# net = FaceModel.FaceModel()
net = DeepEmotionModel.DeepEmotionModel()
learningRate = 0.001
epochs = 100


def trainBenchmark(modelNamePrefix, schedulerChoice, net, times=1):
    """
    测试训练
    :param modelNamePrefix: 输出模型名称的前缀
    :param schedulerChoice: 在这里写选择的衰减函数名称
    :param net: 神经网络的模型
    """
    for i in range(times):
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


trainBenchmark("steplr", "steplr", net, times=5)
trainBenchmark("cosann", "cosann", net, times=5)
trainBenchmark("multisteplr", "multisteplr", net, times=5)
