# 笔记

## 关于损失率

似乎前 10 次适用余弦退火是比较好的,可以很快收敛  
而基本上反弹也是在第七次左右.

后续?

- ~~多级别调整~~ 已确认,详情看过拟合部分

> 提一下不能一味看损失率

## 关于模型

FaceModel 是从网上找到的,基本上就是按照标准卷积模型结构来制作的  
可以考虑 DeepMotion? 但是需要校准一下加载器'

> 这里可以提一下对 DataSet 加载的修改

已经使用了该模型

## 关于过拟合

损失率反增问题十分明显,学习率适用 0.1 还是太大了  
使用 0.001 反而还 TMD 好了,出道即巅峰,正确率直接去到 40%

另外还需要打开随机数据集 下一步考虑

- 适用 73 分成

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

## 关于绘图

需要固定集中方式,做 5 回实验,保存其图表作为素材

## 参考

https://blog.csdn.net/u013249853/article/details/89393982

https://blog.csdn.net/qq_42079689/article/details/102806940

https://www.cnblogs.com/lliuye/p/9471231.html

https://www.youtube.com/watch?v=yN7qfBhfGqs

https://blog.csdn.net/qq_35762060/article/details/110545612

https://blog.csdn.net/qq_43815869/article/details/110562733
