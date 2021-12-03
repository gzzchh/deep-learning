"""
测试各种衰减函数与模型
"""
import pandas as pd
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

        # 画图前保存数据
        lossRateSeries = pd.Series(lossRateList)
        learnRateSeries = pd.Series(learnRateList)
        trainAccSeries = pd.Series(trainAccList)
        testAccSeries = pd.Series(testAccList)
        epochSeries = pd.Series(epochList)
        df = pd.DataFrame()
        # 写到 frame 里
        df["epoch"] = epochSeries
        df["lossRate"] = lossRateSeries
        df["learnRate"] = learnRateSeries
        df["trainAcc"] = trainAccSeries
        df["testAcc"] = testAccSeries
        df.to_csv(f"{modelNamePrefix}-{i}-data.csv", index=False)

        # 把各种参数塞进去画图
        fig, ax = plt.subplots()
        ax.plot(epochList, lossRateList, color="blue", label="轮数")
        ax.plot(epochList, learnRateList, color="orange", label="学习率")
        ax.plot(epochList, trainAccList, color="green", label="训练正确率")
        ax.plot(epochList, testAccList, color="red", label="测试正确率")

        plt.savefig(f"{modelNamePrefix}-{i}.png")
        plt.show()
        torch.save(model, f"{modelNamePrefix}-{i}.pt")


# trainBenchmark("steplr", "steplr", net, times=5)
# trainBenchmark("cosann", "cosann", net, times=5)
# trainBenchmark("multisteplr", "multisteplr", net, times=5)


trainBenchmark("cosann-best", "cosann", net, times=1)
