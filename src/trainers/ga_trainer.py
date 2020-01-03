import os

from torch import nn, optim

from utils.train_constants import *
from dataset.FaceDataset import FaceDataset
from models.evolutionary.face_classifier import FaceClassifier
from trainers.trainer import Trainer, EarlyStoppingException


class GATrainer(Trainer):
    def __init__(self, model: FaceClassifier, dataset: FaceDataset, log_tag: str):
        super().__init__(dataset, log_tag)

        self.model = model

        self.patience = len(self.loader)
        self.worse_iterations = 0
        self.best_val_loss = None

        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=0.001)

    def _run_batch(self, images: torch.Tensor, labels: torch.Tensor, iteration: int = 0) -> None:
        self.model.train()
        self.model.zero_grad()

        labels = labels.float()

        # Labels for regression in [0, 1]
        noise_prob = torch.rand(size=(images.size(0),))
        bernoulli = torch.distributions.Bernoulli(probs=noise_prob)
        train_labels = labels * (torch.ones_like(noise_prob) - noise_prob).to(DEVICE)

        noise = (torch.rand(size=(images.size())) * 2 - 1).to(DEVICE)
        mask = bernoulli.sample(sample_shape=images[0].size()).permute(3, 0, 1, 2).to(DEVICE)
        train_images = torch.where(mask.bool(), noise, images)

        pred = self.model(train_images)
        loss = self.criterion(pred, train_labels)

        loss.backward()
        self.optim.step()

        self.writer.add_image(f'Img_{train_labels[0].cpu()}',
                              (train_images[0].cpu().detach() + 1) / 2,
                              global_step=iteration)

        # Set Network in evaluation mode and calculate validation loss
        # with the same images applying a different random noise.
        self.model.eval()
        with torch.no_grad():
            noise_prob = torch.rand(size=(images.size(0),))
            bernoulli = torch.distributions.Bernoulli(probs=noise_prob)
            val_labels = labels * (torch.ones_like(noise_prob) - noise_prob).to(DEVICE)

            noise = (torch.rand(size=(images.size())) * 2 - 1).to(DEVICE)
            mask = bernoulli.sample(sample_shape=images[0].size()).permute(3, 0, 1, 2).to(DEVICE)
            val_images = torch.where(mask.bool(), noise, images)

            pred = self.model(val_images)
            val_loss = self.criterion(pred, val_labels)

        # Log loss values in Tensorboard.
        self.writer.add_scalars('Loss values',
                                {'Train': loss, 'Validation': val_loss},
                                global_step=iteration)

        # Early Stopping
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.worse_iterations = 0
        else:
            self.worse_iterations += 1
            if self.worse_iterations >= self.patience:
                raise EarlyStoppingException(f'Early Stopping at Iteration {iteration}')

    def _init_model(self):
        # Send network to the corresponding device (GPU or CPU)
        self.model = self.model.to(device=DEVICE)

        # Set network in train mode
        self.model.train()

    def _save_checkpoint(self, epoch: int):
        path = os.path.join(os.environ['CKPT_DIR'], f'GA_{self.log_tag}')
        if not os.path.exists(path):
            os.mkdir(path)

        save_path = os.path.join(path, f'{epoch}.pt')
        torch.save(self.model.state_dict(), save_path)

    def _get_result_sample(self):
        return None
