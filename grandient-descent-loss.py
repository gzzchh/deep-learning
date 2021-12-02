# 用梯度下降实现损失最小
import torch
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
w = torch.tensor([0.], requires_grad=True)


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    l = ((y_pred - y) ** 2).mean()
    if w.grad is not None:
        w.grad.data.zero_()
    l.backward()
    return l.data


def optimize(learningRate):
    w.data -= learningRate * w.grad.data


epoch_list = []
loss_list = []
w_list = []
for epoch in range(100):
    l = loss(x_data, y_data)
    print('epoch:', epoch, 'loss:', l.item())
    epoch_list.append(epoch)
    loss_list.append(l.item())
    w_list.append(w.item())
    optimize(0.05)

# 打印预测结果
print("predict: 4-> %.0f" % forward(4).item())
plt.plot(epoch_list, loss_list)
plt.plot(epoch_list, w_list)
plt.show()
