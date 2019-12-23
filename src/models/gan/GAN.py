from torch import nn

from models.gan.Discriminator import Discriminator
from models.gan.Generator import Generator


class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()

    def forward(self, x):
        pass
