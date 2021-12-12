# 论文

本作品结合 PyTorch 深度学习框架,实现了一个基于 OpenCV 的表情识别系统.

数据集选择了 FER2013,共有 6 类表情,并且

- 一共 28708 张图片
- 其中 70% 用于训练
- 剩下的 30% 用于测试

## 模型概述

选择的模型是 CNN 模型,取自论文 [Deep-Emotion: Facial Expression Recognition Using Attentional Convolutional Network](https://arxiv.org/abs/1704.07486)

该模型比简易 CNN 模型多了一个 Attention 层,使得模型可以更加精准的提取表情.  
并且通过添加仿射变换,来增加学习的素材数量,同时也可以让神经网络对带有简单变换的图片进行学习.  
这部分的代码如下

```py
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
```

对此论文进行解读的视频是 [Realtime Face Emotion Recognition | PyTorch | Python| Deep Emotion | Stepwise Implementation](https://www.youtube.com/watch?v=yN7qfBhfGqs)  
模型部分引用自该视频的代码,而 OpenCV 实践部分参考了该视频的代码.  
使用这篇文章作为整体结构的参考 [Pytorch 基于卷积神经网络的人脸表情识别（白话 CNN+数据集+代码实战）](https://blog.csdn.net/qq_43815869/article/details/110562733)  
对于数据集的解读,可以参考 [FER2013 数据集说明](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

## 参数调优

在训练中参数调优也是很重要的,所以在代码中我收集了以下参数

- epochs: 训练的轮数
- lossRate: 训练的损失率
- learningRate: 训练的学习率
- trainAccuracy: 训练的准确率
- testAccuracy: 测试的准确率

并且总结了一些经验

一开始将学习率设定为 0.1 还是太大了,这导致模型准确率无法上升,始终停留在 24% 左右  
后来设定为 0.001 之后可以在日志中看到较正常的模型准确率上升,但是这个模型的准确率还是不够高,还是需要进一步调优.

接下来的问题还是过拟合,那么就需要想办法衰减学习率.这里选择了在 PyTorch 中预制了几种

- 定步长衰减 (此处采用每 10 轮衰减 0.1) StepLR
- 多间隔衰减 MultiStepLR
- 余弦退火 CosineAnnealingLR
- 参数判断 ReduceLROnPlateau

参数判断,即当某项参数不变时,对学习率进行调整.但是在实践中发现收效甚微,所以最终训练的时候没有使用.  
而剩下三种方式的参数定义如下

```py
# 逐步调整,10 次一减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# 多间隔
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
    # 第一次调整基本上在第五次之后第十次之内
    10,
    20,
    50,
], gamma=0.1)
```

似乎前 10 次适用余弦退火是比较好的,可以很快收敛,而基本上反弹也是在第七次左右.所以将开始衰减的轮数设定为 10.

下面是两个效果较为明显的损失对比

### 10 轮损失

设定了学习率在 10 次以后衰减到 10%  
现在在第 20 次出现损失率反弹,而且后面有较大的损失率反弹,比如说 1.3 直接到 1.7

目前最好的成绩是在 50 轮达到 54%正确率

- 15 轮达到 46%
- 35 轮达到 52%
- 50 轮达到 54%

但是在 10 轮以后增速也降下来了,感觉还能再优化

考虑到在第 20 次之后出现反弹,可以考虑在 20 次再加一次损失,或者干脆设定成 10 轮损失一次

### 余弦退火

目前发现在 10 轮就达到了 46%的正确率,15 轮已经去到 50%

- 10 轮达到 46%
- 15 轮达到 50%
- 25 轮达到 52%
- 35 轮达到 54%
- 40 轮达到 55%
- 50 轮达到 56%
- 95 轮达到 58%
- 100 轮达到 59%

也存在明显的降速,大概发生在 15 轮之后,考虑到这个应该是模型的极限了

损失率在第 3 轮和第 11 轮都有反弹,但是对于后续正确率增长没有很大影响

## 训练结果比对

开发和测试设备是

- AMD Ryzen 9 3900XT
- RTX 2070 MQ
- 48G

随后送去云端进行训练/记录/绘图

- E5-2699v4 8vCore
- P5000
- 32G

每一种跑 5 次,一次 200 轮,并且用 matplotlib 对记录的数据绘图.

通过比较发现余弦退火的效果一直都很好,而 StepLR 每 10 轮衰减一次到最后就不学习了.

## 参考资料

https://blog.csdn.net/u013249853/article/details/89393982

https://blog.csdn.net/qq_42079689/article/details/102806940

https://www.cnblogs.com/lliuye/p/9471231.html

https://www.youtube.com/watch?v=yN7qfBhfGqs

https://blog.csdn.net/qq_35762060/article/details/110545612

https://blog.csdn.net/qq_43815869/article/details/110562733
