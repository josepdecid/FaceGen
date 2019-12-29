from torch import nn

from constants.train_constants import Z_SIZE


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(in_features=Z_SIZE, out_features=512 * 5 * 5)

        self.conv_transposed = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=5, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.batch_norms = [nn.BatchNorm2d(i ** 2) for i in range(8, 6, -1)]

    def forward(self, x):
        x = self.linear(x).view(-1, 512, 5, 5)
        x = self.conv_transposed(x)
        return x
