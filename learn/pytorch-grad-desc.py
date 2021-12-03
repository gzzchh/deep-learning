"""
用上 Pytorch
"""
import torch
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1, False)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []
w_list = []
for epoch in range(100):
    # 计算预测
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    print(epoch, "l=", loss.item(), "w=%.6f" % model.linear.weight.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    w_list.append(model.linear.weight.item())
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 优化模型
    optimizer.step()

# 打印预测结果
plt.plot(epoch_list, loss_list)
plt.plot(epoch_list, w_list)
plt.ylabel('loss + w')
plt.xlabel('epoch')
plt.show()
