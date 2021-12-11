import os

import pandas as pd


def reFormat(filename):
    df = pd.read_csv(f"./data/{filename}")
    # print(df)
    # trainAcc = df["trainAcc"]
    # testAcc = df["testAcc"]
    # 必须取出为 Series 否则会存在顺序问题
    trainAcc = df.trainAcc
    testAcc = df.testAcc
    df["trainAccFixed"] = [None] * 200
    df["testAccFixed"] = [None] * 200
    for index, row in df.iterrows():
        # 按照 5 次 1 格重新填充 trainAcc 和 testAcc
        accIndex = int((index + 1) / 5) - 1
        # if (index + 1) % 5 == 0:
        #     print(f"次数 {index + 1} 索引 {(index + 1) / 5} 值 {trainAcc.iloc[accIndex]} {testAcc.iloc[accIndex]}")
        # row["trainAccFixed"] = trainAcc.iloc[accIndex]*100
        # row["testAccFixed"] = testAcc.iloc[accIndex]*100   
        row["trainAccFixed"] = trainAcc.iloc[index] * 100
        row["testAccFixed"] = testAcc.iloc[index] * 100
        # 转换数组,只要第一个
        row["learnRate"] = str(row["learnRate"]).replace("[", "").replace("]", "")
        df.iloc[index] = row

    # 删掉原来的 trainAcc 和 testAcc
    df.drop(["trainAcc", "testAcc"], axis=1, inplace=True)
    # print(df)
    # 重命名新的 trainAccFixed 和 testAccFixed
    df.rename(columns={"trainAccFixed": "trainAcc", "testAccFixed": "testAcc"}, inplace=True)
    # 写入新的 csv 文件
    df.to_csv(f"./data-fixed/{filename}", index=False)


# 确保 data-fixed 目录存在
if not os.path.exists("./data-fixed"):
    os.mkdir("./data-fixed")

files = os.listdir("./data")
for file in files:
    # 检测后缀名是否为 csv
    if file.endswith(".csv"):
        # print(file)
        reFormat(file)
