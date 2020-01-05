import os

import torch
from torch import optim, nn

from dataset.FaceDataset import FaceDataset
from models.autoencoder.vae import VAE
from models.autoencoder.vae_loss import MSEKLDLoss
from trainers.trainer import Trainer, EarlyStoppingException
from utils.train_constants import DEVICE, VAE_Z_SIZE

import torchvision.utils as vutils


class VAETrainer(Trainer):
    def __init__(self, model: VAE, log_tag: str, train_dataset: FaceDataset, val_dataset: FaceDataset = None):
        super().__init__(log_tag, train_dataset, val_dataset)

        self.model = model
        self.criterion = MSEKLDLoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=0.002)

        self.patience = 50
        self.best_val_loss = None

    def _run_batch(self, images: torch.Tensor, labels: torch.Tensor,
                   val_images: torch.Tensor = None, val_labels: torch.Tensor = None, iteration: int = 0) -> None:
        # Set model in Train model and clear gradients
        self.model.train()
        self.model.zero_grad()

        # Forward data to the model, calculate loss, backpropagate and optimize the parameters.
        reconstructed_images, mu, log_var = self.model(images)
        loss = self.criterion(reconstructed_images, images, mu, log_var)
        loss.backward()
        self.optim.step()

        # Calculate validation data loss
        self.model.eval()
        with torch.no_grad():
            reconstructed_images, mu, log_var = self.model(val_images)
            val_loss = self.criterion(reconstructed_images, val_images, mu, log_var)

        # Log loss values in Tensorboard
        self.writer.add_scalars('Loss values',
                                {'Train': loss, 'Validation': val_loss},
                                global_step=iteration)

        # Early Stopping
        # if self.best_val_loss is None or val_loss < self.best_val_loss:
        #     self.best_val_loss = val_loss
        #     self.worse_iterations = 0
        # else:
        #     self.worse_iterations += 1
        #     if self.worse_iterations >= self.patience:
        #         raise EarlyStoppingException(f'Early Stopping at Iteration {iteration} ({iteration - self.patience})')

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

    def _get_result_sample(self, iteration: int = 0):
        self.model.eval()
        with torch.no_grad():
            # Generate 8 Reconstructions
            real_samples = torch.stack([self.val_dataset[i][0] for i in range(8)]).to(DEVICE)
            reconstructed_samples, _, _ = self.model(real_samples)

            real_samples = (real_samples.cpu() + 1) / 2
            reconstructed_samples = (reconstructed_samples.cpu() + 1) / 2

            reconstructed_grid = vutils.make_grid(torch.cat([real_samples, reconstructed_samples]), padding=2, nrow=8)
            self.writer.add_image(f'Reconstructed Validation Images',
                                  img_tensor=reconstructed_grid, global_step=iteration)

            # Generate 12 random samples
            latent = torch.randn(size=(12, VAE_Z_SIZE)).to(DEVICE)
            output = self.model.decode(latent)
            fake_samples = (output.cpu() + 1) / 2

            fake_grid = vutils.make_grid(fake_samples, padding=2, nrow=4)
            self.writer.add_image(f'Generated Samples',
                                  img_tensor=fake_grid, global_step=iteration)

    def _save_checkpoint(self, epoch: int):
        path = os.path.join(os.environ['CKPT_DIR'], f'VAE_{self.log_tag}')
        if not os.path.exists(path):
            os.mkdir(path)

        save_path = os.path.join(path, f'{epoch}.pt')
        torch.save(self.model.state_dict(), save_path)
