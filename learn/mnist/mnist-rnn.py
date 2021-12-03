"""
使用MNIST进行RNN
"""
import torch
from torch import nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class RNNNet(nn.Module):
    def __init__(self):
        super(RNNNet, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        out, out_h = self.rnn(x, None)
        out = self.fc(out[:, -1, :])
        return out


rnnnet = RNNNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnnnet.parameters())

# 现在开始训练模型
n_epochs = 10
for epoch in range(n_epochs):
    running_loss = 0.0
    running_corrects = 0
    print(f"Epoch {epoch}/{n_epochs}")
    print("." * 10)
    for i, (X, y) in enumerate(trainloader):
        X, y = X.to(device), y.to(device)
        X = X.view(-1, 28, 28)
        y_out = rnnnet(X)
        loss = criterion(y_out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        value, y_pred = torch.max(y_out.data, 1)
        running_corrects += torch.sum(y_pred.data == y.data)

    teseting_correct = 0
    for i, (Xt, yt) in enumerate(testloader):
        Xt = Xt.to(device)
        yt = yt.to(device)
        Xt = Xt.view(-1, 28, 28)
        yt_out = rnnnet(Xt)
        value, yt_pred = torch.max(yt_out.data, 1)
        teseting_correct += torch.sum(yt_pred.data == yt.data)

    print(f"""
    Loss is {running_loss / len(trainset)}, 
    训练正确性: {100 * running_corrects / len(trainset)}
    测试正确性: {100 * teseting_correct / len(testset)}
    """)

# 保存模型
torch.save(rnnnet.state_dict(), 'rnn_model')
