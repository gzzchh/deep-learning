import os

import cv2
import numpy as np
import pandas as pd


def mkdirIfNotExist(path):
    """
    创建文件夹
    :param path: 
    :return: 
    """
    if not os.path.exists(path):
        os.mkdir(path)


def extractTrainSet():
    """
    将 csv 中的纯数字转换为图片
    用于提取训练集
    :return: 
    """
    # 数据预处理
    # 准备目录
    mkdirIfNotExist("datasets-labeled/train")
    mkdirIfNotExist("datasets-labeled/test")
    # 将label与人脸数据作拆分
    df = pd.read_csv("datasets-raw/train.csv")
    # 获取df行数
    linesCount = len(df.index)
    # 确保都有一个值
    df = df.fillna(0)
    # 定义训练占比
    trainPercent = 0.7
    trainCount = int(linesCount * trainPercent)
    # 为分类准备列表
    trainLabel, testLabel = [], []
    trainPath, testPath = [], []
    for index, row in df.iterrows():
        emotionLabel, pixels = row["emotion"], row["pixels"]
        pixels = pixels.split(" ")
        # print(pixels)
        face_array = np.array(pixels, dtype=int).reshape((48, 48))
        # 决定是否是训练集
        if index < trainCount:
            cv2.imwrite(f"datasets-labeled/train/{index}.jpg", face_array)
            trainPath.append(f"train/{index}.jpg")
            trainLabel.append(emotionLabel)
        else:
            cv2.imwrite(f"datasets-labeled/test/{index}.jpg", face_array)
            testPath.append(f"test/{index}.jpg")
            testLabel.append(emotionLabel)

    trainLabelSeries = pd.Series(trainLabel)
    trainPathSeries = pd.Series(trainPath)
    testLabelSeries = pd.Series(testLabel)
    testPathSeries = pd.Series(testPath)
    dfTrain = pd.DataFrame(
        {"label": trainLabelSeries, "path": trainPathSeries})
    dfTest = pd.DataFrame({"label": testLabelSeries, "path": testPathSeries})
    dfTrain.to_csv("datasets-labeled/train.csv", index=False, header=False)
    dfTest.to_csv("datasets-labeled/test.csv", index=False, header=False)


def labelImages(path):
    """
    将图片标记和文件名关联起来
    :param path: 输入路径
    :return: 
    """
    # 读取label文件
    dfLabel = pd.read_csv("datasets-unkown/train-label.csv", header=None)
    # 准备数据
    pics = os.listdir(path)
    picPathList = []
    labelList = []
    # 现在来做标记
    for pic in pics:
        if os.path.splitext(pic)[1] == ".jpg":
            # 首先你得是一个图片
            # 名称加进来,比如说 1.jpg
            picPathList.append(pic)
            # 找出文件名,转换成数字 1
            filename = int(os.path.splitext(pic)[0])
            # 找出对应的label 比如说 1.jpg 对应的label是 0
            # 那么就填入
            labelList.append(dfLabel.iat[filename, 0])
    # 做成 Series
    picPathSeries = pd.Series(picPathList)
    labelSeries = pd.Series(labelList)
    df = pd.DataFrame()
    # 写到 frame 里
    df["path"] = picPathSeries
    df["label"] = labelSeries
    df.to_csv(f"{path}/train-dataset.csv", index=False, header=False)


extractTrainSet()
