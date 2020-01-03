import os
import torch
from torch import optim, nn

from utils.train_constants import DEVICE, Z_SIZE
from dataset.FaceDataset import FaceDataset
from models.autoencoder.vae import VAE
from models.autoencoder.vae_loss import MSEKLDLoss
from trainers.trainer import Trainer


class VAETrainer(Trainer):
    def __init__(self, model: VAE, dataset: FaceDataset, log_tag: str):
        super().__init__(dataset, log_tag)

        self.model = model
        self.criterion = MSEKLDLoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=0.002)

    def _run_batch(self, images: torch.Tensor, labels: torch.Tensor = None, iteration: int = 0) -> None:
        self.model.zero_grad()

        reconstructed_images, mu, log_var = self.model(images)

        loss = self.criterion(reconstructed_images, images, mu, log_var)
        loss.backward()

        self.optim.step()

        self.writer.add_scalar('Loss', loss, iteration)

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

    def _get_result_sample(self):
        self.model.eval()
        with torch.no_grad():
            latent = torch.randn(size=(9, Z_SIZE)).to(DEVICE)
            output = self.model.decode(latent).cpu()
            fake_images = (output - (-1)) / (1 - (-1))
        self.model.train()
        return fake_images

    def _save_checkpoint(self, epoch: int):
        save_path = os.path.join(os.environ["CKPT_DIR"], f'VAE_{self.log_tag}_{epoch}.pt')
        torch.save(self.model.state_dict(), save_path)
