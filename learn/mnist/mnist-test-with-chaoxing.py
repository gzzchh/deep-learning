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
fcnet.load_state_dict(torch.load("./myfcnet_model"))

i = iter(testloader)
images, labels = next(i)
images = images.to(device)
labels = labels.to(device)
y_out = fcnet(images)
value, y_pred = torch.max(y_out.data, 1)
print(y_pred.data)

img = torchvision.utils.make_grid(images)
img = img.to(torch.device("cpu")).numpy().transpose(1, 2, 0)
plt.imshow(img)
plt.show()
