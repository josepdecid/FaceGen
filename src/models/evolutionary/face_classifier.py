from torch import nn


class FaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=2048, out_features=100),
            nn.ReLU(),

            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),

            nn.Linear(in_features=50, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b_size = x.size(0)
        x = self.conv(x)
        x = x.view(b_size, -1)
        x = self.linear(x)
        return x.view(-1)
