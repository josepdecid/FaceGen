from random import random

import os
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from constants.train_constants import *
from dataset.UTKFaceDataset import UTKFaceDataset
from models.gan.GAN import GAN

from torch.utils.tensorboard import SummaryWriter


def _run_batch():
    pass


def _run_epoch(model: GAN, loader: DataLoader, writer: SummaryWriter, epoch_idx: int):
    num_batches = len(loader)
    for batch_idx, images in enumerate(tqdm(loader,
                                            total=num_batches, ncols=150,
                                            position=0, leave=True,
                                            desc=f'Epoch {epoch_idx:4}')):
        _run_batch()
        writer.add_scalar('Generator Loss', random(), epoch_idx * BATCH_SIZE + batch_idx)
        writer.add_scalar('Discriminator Loss', random(), epoch_idx * BATCH_SIZE + batch_idx)


def train(model: GAN, dataset: UTKFaceDataset):
    writer = SummaryWriter(log_dir=os.environ['LOG_DIR'])

    loader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=SHUFFLE_TRAIN)

    for epoch_idx in range(EPOCHS):
        writer.add_image(f'Epoch {epoch_idx} generation',
                         img_tensor=dataset[0],
                         global_step=epoch_idx)

        _run_epoch(model, loader, writer, epoch_idx, )

    writer.close()
