from torch import nn


class FaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Dropout(),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=30976, out_features=100),
            nn.ReLU(),

            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),

            nn.Linear(in_features=50, out_features=1),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        b_size = x.size(0)
        x = self.conv(x)
        x = x.view(b_size, -1)
        x = self.linear(x)
        return x.view(-1)
