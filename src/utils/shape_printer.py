from torch import nn


class ShapePrinter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(x.size())
        return x
