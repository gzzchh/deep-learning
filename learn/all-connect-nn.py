import torch
# 一次输入数据量
batch_n = 100
# 特征数量
input_data = 1000
# 隐层数量
hidden_layer = 100
# 输出层数据量
output_data = 10
# 训练次数
epoch_n = 150
# 学习速度
learning_rate = 1e-6

# 建立全连接神经网络
x = torch.randn(batch_n, input_data)
y = torch.randn(batch_n, output_data)
w1 = torch.randn(input_data, hidden_layer)
w2 = torch.randn(hidden_layer, output_data)

for epoch in range(epoch_n):
    h1 = x.mm(w1)
    h1 = h1.clamp(min=0)
    y_pred = h1.mm(w2)
    # 计算 100 x 10 的损失
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, loss:{},".format(epoch, loss.item()))
    # 梯度下降
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h1.t().mm(grad_y_pred)

    grad_h = grad_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp_(min=0)
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
