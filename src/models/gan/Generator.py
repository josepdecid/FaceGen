import torch
from torch import nn
from torch.nn import functional as F

from constants.train_constants import Z_SIZE


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(in_features=Z_SIZE, out_features=512 * 5 * 5)

        self.conv_layers = [
            nn.ConvTranspose2d(in_channels=512, out_channels=256,
                               kernel_size=4, stride=5, output_padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=3,
                               kernel_size=4, stride=2, padding=1)
        ]

        self.batch_norms = [nn.BatchNorm2d(i ** 2) for i in range(8, 6, -1)]

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 512, 5, 5)

        for i in range(len(self.conv_layers) - 1):
            print(x.size())
            x = self.conv_layers[i](x)
            x = F.relu(x)

        x = self.conv_layers[-1](x)
        return torch.tanh(x)
