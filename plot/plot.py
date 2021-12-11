"""
读取各种数据,开始绘制图表
"""
import os

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

import pandas as pd


def plot(filename):
    df = pd.read_csv(f"./data-fixed/{filename}")

    epoch = df.epoch
    lossRate = df.lossRate
    learnRate = df.learnRate
    trainAcc = df.trainAcc
    testAcc = df.testAcc

    figure, axes = plt.subplots(3, 1, constrained_layout=True)

    # 第一张图,绘制lossRate
    axes[0].plot(epoch, lossRate, color="blue", label="lossRate")
    axes[0].set_title("lossRate")
    # 第二张图,绘制learnRate
    axes[1].plot(epoch, learnRate, color="orange", label="learnRate")
    axes[1].set_title("learnRate")
    # 第三张图,绘制trainAcc和testAcc
    axes[2].plot(epoch, trainAcc, color="green", label="trainAcc")
    axes[2].plot(epoch, testAcc, color="red", label="testAcc")
    axes[2].set_title("train/test Acc")
    # 从 filename 获取文件名
    basename = os.path.basename(filename)
    plt.savefig(f"./plots/{basename}.png")
    plt.show()


# 确保 plots 目录存在
if not os.path.exists("./plots"):
    os.mkdir("./plots")

files = os.listdir("./data-fixed")
for file in files:
    # 检测后缀名是否为 csv
    if file.endswith(".csv"):
        # print(file)
        plot(file)
