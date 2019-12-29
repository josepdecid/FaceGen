from torch import nn, optim

from constants.train_constants import *
from dataset.UTKFaceDataset import UTKFaceDataset
from models.evolutionary.face_classifier import FaceClassifier
from trainers.trainer import Trainer


class GATrainer(Trainer):
    def __init__(self, model: FaceClassifier, dataset: UTKFaceDataset, log_tag: str):
        super().__init__(dataset, log_tag)

        self.model = model

        self.criterion = nn.BCELoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=0.01)

    def _run_batch(self, images, iteration):
        b_size = images.size(0)
        labels = torch.full(size=(b_size,), fill_value=1, device=DEVICE)

        self.model.zero_grad()

        pred = self.model(images)
        loss = self.criterion(pred, labels)

        loss.backward()

        self.optim.step()

        self.writer.add_scalar(f'({self.log_tag}) Loss', loss, iteration)

    def _init_model(self):
        # Send network to the corresponding device (GPU or CPU)
        self.model = self.model.to(device=DEVICE)

        # Set network in train mode
        self.model.train()

    def _get_result_sample(self):
        return None
