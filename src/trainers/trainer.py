import os
from abc import abstractmethod, ABC

import torchvision.utils as vutils
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from utils.train_constants import *


class Trainer(ABC):
    def __init__(self, log_tag: str, train_dataset: VisionDataset, val_dataset: VisionDataset = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        if self.val_dataset is not None:
            self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)

        self.writer = SummaryWriter(log_dir=os.environ['LOG_DIR'])
        self.log_tag = log_tag

    def train(self):
        self._init_model()

        for epoch_idx in range(EPOCHS):
            try:
                self._run_epoch(epoch_idx=epoch_idx)
                self._save_checkpoint(epoch=epoch_idx)
            except EarlyStoppingException as e:
                print(f'Early Stopping at iteration {e.message} (epoch {epoch_idx})')
                self._save_checkpoint(epoch=epoch_idx)
                break

        self.writer.close()

    def _run_epoch(self, epoch_idx: int):
        num_batches = len(self.train_loader)
        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader,
                                                          total=num_batches, ncols=100,
                                                          position=0, leave=True,
                                                          desc=f'Epoch {epoch_idx:4}')):
            iteration = epoch_idx * num_batches + batch_idx

            if self.val_dataset:
                val_images, val_labels = next(iter(self.val_loader))
                self._run_batch(images.to(DEVICE), labels.to(DEVICE),
                                val_images.to(DEVICE), val_labels.to(DEVICE),
                                iteration=iteration)
            else:
                self._run_batch(images.to(DEVICE), labels.to(DEVICE),
                                iteration=iteration)

            if batch_idx % 100 == 0:
                fake_samples = self._get_result_sample()
                if fake_samples is not None:
                    fake_grid = vutils.make_grid(fake_samples, padding=2, nrow=3)
                    self.writer.add_image(f'Iteration {iteration} generation',
                                          img_tensor=fake_grid, global_step=iteration)

    @abstractmethod
    def _run_batch(self, images: torch.Tensor, labels: torch.Tensor,
                   val_images: torch.Tensor = None, val_labels: torch.Tensor = None, iteration: int = 0) -> None:
        pass

    @abstractmethod
    def _get_result_sample(self):
        pass

    @abstractmethod
    def _init_model(self):
        pass

    @abstractmethod
    def _save_checkpoint(self, epoch: int):
        pass


class EarlyStoppingException(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, iteration):
        self.message = iteration
