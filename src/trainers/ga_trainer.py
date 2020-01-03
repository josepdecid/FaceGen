import os

from torch import nn, optim

from constants.train_constants import *
from dataset.UTKFaceDataset import UTKFaceDataset
from models.evolutionary.face_classifier import FaceClassifier
from trainers.trainer import Trainer


class GATrainer(Trainer):
    def __init__(self, model: FaceClassifier, dataset: UTKFaceDataset, log_tag: str):
        super().__init__(dataset, log_tag)

        self.model = model

        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=0.001)

    def _run_batch(self, images: torch.Tensor, labels: torch.Tensor = None, iteration: int = 0) -> None:
        self.model.train()
        self.model.zero_grad()

        noise = torch.rand(size=(images.size(0),)) * 2 - 1
        train_images = images + torch.tensordot(a=noise.view(-1, 1),
                                                b=torch.rand(size=images.size()),
                                                dims=1).to(DEVICE)
        labels = (torch.ones_like(noise) - torch.abs(noise)).to(DEVICE)

        pred = self.model(train_images)
        loss = self.criterion(pred, labels)

        loss.backward()
        self.optim.step()

        # self.writer.add_image(f'Img_{labels[0].cpu()}',
        #                       (images[0].cpu().detach() + 1) / 2,
        #                       global_step=iteration)

        # Set Network in evaluation mode and calculate validation loss
        # with the same images applying a different random noise.
        self.model.eval()
        with torch.no_grad():
            noise = torch.rand(size=(images.size(0),)) * 2 - 1
            val_images = images + torch.tensordot(a=noise.view(-1, 1),
                                                  b=torch.rand(size=images.size()),
                                                  dims=1).to(DEVICE)
            labels = (torch.ones_like(noise) - torch.abs(noise)).to(DEVICE)

            pred = self.model(val_images)
            val_loss = self.criterion(pred, labels)

        # Log loss values in Tensorboard.
        self.writer.add_scalars('Loss values',
                                {'Train': loss, 'Validation': val_loss},
                                global_step=iteration)

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
