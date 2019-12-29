import torch
from torch import optim, nn

from constants.train_constants import DEVICE
from dataset.UTKFaceDataset import UTKFaceDataset
from models.autoencoder.vae import VAE
from models.autoencoder.vae_loss import MSEKLDLoss
from trainers.trainer import Trainer


class VAETrainer(Trainer):
    def __init__(self, model: VAE, dataset: UTKFaceDataset, log_tag: str):
        super().__init__(dataset, log_tag)

        self.model = model
        self.criterion = MSEKLDLoss()
        self.optim = optim.Adam(params=self.model.parameters())

    def _run_batch(self, images: torch.Tensor, iteration: int) -> None:
        b_size = images.size(0)

        self.model.zero_grad()

        reconstructed_images, mu, log_var = self.model(images)

        loss = self.criterion(reconstructed_images, images, mu, log_var)
        loss.backward()

        self.optim.step()

        self.writer.add_scalar('Loss', loss, iteration, tag=self.log_tag)

    def _init_model(self):
        # Send network to the corresponding device (GPU or CPU)
        self.model = self.model.to(device=DEVICE)

        # Set network in train mode
        self.model.train()

        # Initialize weights
        self.model.apply(VAETrainer.__weights_init)

    @staticmethod
    def __weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
