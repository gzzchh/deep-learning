import numpy as np
import matplotlib.pyplot as plt

# 这是三个点的数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 假定线性模型 y = w * x


def forward(x):
    return x*w


# 计算损失函数, 此处用平方差,也就是距离作为损失函数
def loss(x, y):
    # 这算是函数表达式
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []
# 枚举不同的斜率
for w in np.arange(0, 5, 1):
    print("w:", w)
    l_sum = 0
    # 每一次都要拿三个点来算,所以这里用for循环
    for x_val, y_val in zip(x_data, y_data):
        # 在这个 x 下他的表达式为 y_pred
        y_pred_val = forward(x_val)
        # 此时计算对应的损失函数
        l = loss(x_val, y_val)
        # 最后加起来
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    # 但是别忘了这是加起来的,所以这里除以了 3
    print("MSE:", l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.xlabel("w")
plt.ylabel("MSE")
# plt.xlable("w")
plt.show()
