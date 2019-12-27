import torch
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.dense1 = nn.Linear(in_features=73728, out_features=1000)
        self.dense2 = nn.Linear(in_features=1000, out_features=100)
        self.dense3 = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)

        x = x.view(x.size(0), -1)

        x = self.dense1(x)
        x = F.leaky_relu(x)

        x = self.dense2(x)
        x = F.leaky_relu(x)

        x = self.dense3(x)
        return torch.sigmoid(x)
