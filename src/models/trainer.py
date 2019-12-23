from torch.utils.data.dataloader import DataLoader

from constants.train_constants import *
from models.gan.GAN import GAN


def _run_batch():
    pass


def _run_epoch(model: GAN, loader: DataLoader):
    for batch_idx in enumerate(loader):
        _run_batch()


def train(model: GAN, loader: DataLoader):
    for epoch_idx in range(EPOCHS):
        _run_epoch(model, loader)
