import time
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from constants.train_constants import *
from dataset.UTKFaceDataset import UTKFaceDataset
from models.gan.GAN import GAN


def _run_batch():
    pass


def _run_epoch(model: GAN, loader: DataLoader, epoch_idx):
    num_batches = len(loader)
    for batch_idx, images in enumerate(tqdm(loader,
                                            total=num_batches, ncols=150,
                                            position=0, leave=True,
                                            desc=f'Epoch {epoch_idx:4}')):
        _run_batch()

    print(f'Generator Loss:     {1234:.6f}')
    print(f'Discriminator Loss: {1234:.6f}')


def train(model: GAN, dataset: UTKFaceDataset):
    loader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=SHUFFLE_TRAIN)

    for epoch_idx in range(EPOCHS):
        _run_epoch(model, loader, epoch_idx)
