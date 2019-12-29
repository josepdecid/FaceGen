import os
from abc import abstractmethod, ABC

import torchvision.utils as vutils
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from constants.train_constants import *


class Trainer(ABC):
    def __init__(self, dataset: VisionDataset, log_tag: str):
        self.dataset = dataset
        self.loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)

        self.writer = SummaryWriter(log_dir=os.environ['LOG_DIR'])
        self.log_tag = log_tag

    def train(self):
        self._init_model()

        for epoch_idx in range(EPOCHS):
            fake_samples = self._get_result_sample()
            if fake_samples is not None:
                fake_grid = vutils.make_grid(fake_samples, padding=2, nrow=3)
                self.writer.add_image(f'Epoch {epoch_idx} generation',
                                      img_tensor=fake_grid,
                                      global_step=epoch_idx)
            self._run_epoch(epoch_idx=epoch_idx)

        self.writer.close()

    def _run_epoch(self, epoch_idx: int):
        num_batches = len(self.loader)
        for batch_idx, data in enumerate(tqdm(self.loader,
                                              total=num_batches, ncols=100,
                                              position=0, leave=True,
                                              desc=f'Epoch {epoch_idx:4}')):
            if isinstance(data, list):
                self._run_batch(data[0].to(DEVICE), data[1].to(DEVICE),
                                iteration=epoch_idx * num_batches + batch_idx)
            else:
                self._run_batch(data.to(DEVICE), torch.zeros(),
                                iteration=epoch_idx * num_batches + batch_idx)

    @abstractmethod
    def _run_batch(self, images: torch.Tensor, labels: torch.Tensor, iteration: int) -> None:
        pass

    @abstractmethod
    def _get_result_sample(self):
        pass

    @abstractmethod
    def _init_model(self):
        pass
