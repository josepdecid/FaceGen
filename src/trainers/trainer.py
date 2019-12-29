import os
from abc import abstractmethod, ABC

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants.train_constants import *
from dataset.UTKFaceDataset import UTKFaceDataset


class Trainer(ABC):
    def __init__(self, dataset: UTKFaceDataset, log_tag: str):
        self.dataset = dataset
        self.loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN)

        self.writer = SummaryWriter(log_dir=os.environ['LOG_DIR'])
        self.log_tag = log_tag

    def train(self):
        self._init_model()

        for epoch_idx in range(EPOCHS):
            self.writer.add_image(f'Epoch {epoch_idx} generation', img_tensor=self.dataset[0], global_step=epoch_idx)
            self._run_epoch(epoch_idx=epoch_idx)

        self.writer.close()

    def _run_epoch(self, epoch_idx: int):
        num_batches = len(self.loader)
        for batch_idx, images in enumerate(tqdm(self.loader,
                                                total=num_batches, ncols=150,
                                                position=0, leave=True,
                                                desc=f'Epoch {epoch_idx:4}')):
            self._run_batch(images, iteration=epoch_idx * BATCH_SIZE + batch_idx)

    @abstractmethod
    def _run_batch(self, images: torch.Tensor, iteration: int) -> None:
        pass

    @abstractmethod
    def _init_model(self):
        pass
