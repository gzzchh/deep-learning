import torch
import torchvision
from torchvision import transforms, datasets
from matplotlib import pyplot as plt

# 使用 CUDA 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备转换层
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

# 准备训练集
trainset = datasets.MNIST(root='./data', train=True,
                          download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, )

# 准备测试集
testset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=5, shuffle=True)


class FcNet(torch.nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


fcnet = FcNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fcnet.parameters(), lr=0.001)

n_epoch = 5
for epoch in range(n_epoch):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epoch))
    print("." * 10)
    for i, (x, y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device)
        y_out = fcnet(x)
        loss = criterion(y_out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        value, y_pred = torch.max(y_out.data, 1)
        running_correct += torch.sum(y_pred.data == y.data)

    print("Loss in {:.4f}, Train Accuracy is : {:.4f}%".format(
        running_loss / len(trainset), 100 * running_correct / len(trainset)))

torch.save(fcnet.state_dict(), "./myfcnet_model")
