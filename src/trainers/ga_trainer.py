import os
from torch import nn, optim

from constants.train_constants import *
from dataset.FaceNoFaceDataset import FaceNoFaceDataset
from models.evolutionary.face_classifier import FaceClassifier
from trainers.trainer import Trainer


class GATrainer(Trainer):
    def __init__(self, model: FaceClassifier, dataset: FaceNoFaceDataset, log_tag: str):
        super().__init__(dataset, log_tag)

        self.model = model

        self.criterion = nn.BCELoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=0.01)

    def _run_batch(self, images, labels, iteration):
        self.model.zero_grad()

        pred = self.model(images)
        loss = self.criterion(pred, labels.type(torch.float))

        loss.backward()
        self.optim.step()

        accuracy = 100 * (pred > 0.5).eq(labels).sum() / images.size(0)

        self.writer.add_scalar(f'({self.log_tag}) Loss', loss, iteration)
        self.writer.add_scalar(f'({self.log_tag}) Accuracy', accuracy, iteration)

    def _init_model(self):
        # Send network to the corresponding device (GPU or CPU)
        self.model = self.model.to(device=DEVICE)

        # Set network in train mode
        self.model.train()

    def _save_checkpoint(self, epoch: int):
        save_path = os.path.join(os.environ['CKPT_DIR'], f'{self.log_tag}_{epoch}.pt')
        torch.save(self.model.state_dict(), save_path)

    def _get_result_sample(self):
        return None
