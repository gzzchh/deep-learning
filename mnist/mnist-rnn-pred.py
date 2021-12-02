"""
使用MNIST进行RNN
"""
import matplotlib.pyplot as plt
import torch
import torchvision.utils
from torch import nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

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
rnnnet.load_state_dict(torch.load("rnn_model"))

i = iter(testloader)
images, labels = next(i)
images = images.to(device)
labels = labels.to(device)
x_in = images.view(-1, 28, 28)
y_out = rnnnet(x_in)
value, yr_pred = torch.max(y_out.data, 1)
print(yr_pred.data)

img = torchvision.utils.make_grid(images)
img = img.to(torch.device("cpu")).numpy().transpose(1, 2, 0)
plt.imshow(img)
plt.show()
