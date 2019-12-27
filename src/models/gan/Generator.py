from torch import nn
from torch.nn import functional as F

from constants.train_constants import Z_SIZE


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_t1 = nn.ConvTranspose2d(in_channels=Z_SIZE, out_channels=512,
                                          kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv_t2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                          kernel_size=4, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv_t3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=4, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv_t4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=4, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv_t5 = nn.ConvTranspose2d(in_channels=64, out_channels=3,
                                          kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv_t1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv_t2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv_t3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv_t4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv_t5(x)
        return F.tanh(x)
