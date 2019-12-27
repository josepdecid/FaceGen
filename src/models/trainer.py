import os

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants.train_constants import *
from dataset.UTKFaceDataset import UTKFaceDataset
from models.gan.Discriminator import Discriminator
from models.gan.Generator import Generator


class Trainer:
    def __init__(self, G: Generator, D: Discriminator, dataset: UTKFaceDataset):
        self.G = G
        self.D = D

        self.dataset = dataset
        self.loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)

        self.writer = SummaryWriter(log_dir=os.environ['LOG_DIR'])

        self.criterion = nn.BCELoss()
        self.optim_G: optim.Adam = None
        self.optim_D: optim.Adam = None

    def train(self):
        # Send both networks to the corresponding device (GPU or CPU)
        self.G = self.G.to(device=DEVICE)
        self.D = self.D.to(device=DEVICE)

        # Set both networks in train mode
        self.G.train()
        self.D.train()

        # Initialize weights of both networks
        self.G.apply(Trainer.__weights_init)
        self.D.apply(Trainer.__weights_init)

        for epoch_idx in range(EPOCHS):
            self.writer.add_image(f'Epoch {epoch_idx} generation', img_tensor=self.dataset[0], global_step=epoch_idx)
            self.__run_epoch(epoch_idx=epoch_idx)

        self.writer.close()

    def __run_epoch(self, epoch_idx: int):
        num_batches = len(self.loader)
        for batch_idx, images in enumerate(tqdm(self.loader,
                                                total=num_batches, ncols=150,
                                                position=0, leave=True,
                                                desc=f'Epoch {epoch_idx:4}')):
            err_G, err_D = self.__run_batch(images)
            self.writer.add_scalar('Generator Loss', err_G, epoch_idx * BATCH_SIZE + batch_idx)
            self.writer.add_scalar('Discriminator Loss', err_D, epoch_idx * BATCH_SIZE + batch_idx)

    def __run_batch(self, images):
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
        noise = torch.randn(b_size, Z_SIZE, 1, 1, device=DEVICE)
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

        pred = self.D(fake_images).view(-1)
        err_G = self.criterion(pred, labels)
        err_G.backward()

        self.optim_G.step()

        return err_G, err_D

    @staticmethod
    def __weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
