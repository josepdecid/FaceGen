from torch import nn, optim

from constants.train_constants import *
from dataset.UTKFaceDataset import UTKFaceDataset
from models.gan.Discriminator import Discriminator
from models.gan.Generator import Generator
from trainers.trainer import Trainer


class GANTrainer(Trainer):
    def __init__(self, G: Generator, D: Discriminator, dataset: UTKFaceDataset, log_tag: str):
        super().__init__(dataset, log_tag)

        self.G = G
        self.D = D

        self.criterion = nn.BCELoss()
        self.optim_G: optim.Adam = optim.Adam(params=self.G.parameters())
        self.optim_D: optim.Adam = optim.Adam(params=self.D.parameters())

    def _run_batch(self, images, iteration):
        b_size = images.size(0)

        ################
        # D Optimization
        ################

        self.D.zero_grad()

        labels = torch.full(size=(b_size,), fill_value=1, device=DEVICE)
        pred = self.D(images).view(-1)
        err_real_D = self.criterion(pred, labels)
        err_real_D.backward()

        labels.fill_(value=0)
        noise = torch.randn(size=(b_size, Z_SIZE), device=DEVICE)
        fake_images = self.G(noise)
        pred = self.D(fake_images.detach()).view(-1)
        err_fake_D = self.criterion(pred, labels)
        err_fake_D.backward()

        err_D = err_real_D + err_fake_D

        self.optim_D.step()

        ################
        # G Optimization
        ################

        self.G.zero_grad()

        labels.fill_(value=1)
        pred = self.D(fake_images).view(-1)
        err_G = self.criterion(pred, labels)
        err_G.backward()

        self.optim_G.step()

        self.writer.add_scalar('Generator Loss', err_G, iteration, tag=self.log_tag)
        self.writer.add_scalar('Discriminator Loss', err_D, iteration, tag=self.log_tag)

    def _init_model(self):
        # Send both networks to the corresponding device (GPU or CPU)
        self.G = self.G.to(device=DEVICE)
        self.D = self.D.to(device=DEVICE)

        # Set both networks in train mode
        self.G.train()
        self.D.train()

        # Initialize weights of both networks
        self.G.apply(GANTrainer.__weights_init)
        self.D.apply(GANTrainer.__weights_init)

    @staticmethod
    def __weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
