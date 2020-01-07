import os
import shutil

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

from dataset.FaceDataset import FaceDataset
from models.gan.Generator import Generator
from utils.train_constants import DEVICE, BATCH_SIZE, GAN_Z_SIZE

CHECKPOINT = '96_G.pt'

load_dotenv()

dataset = FaceDataset(os.environ['DATASET_PATH'])
num_samples = len(dataset)

model: Generator = Generator()
model_weights = torch.load(os.path.join(os.environ['CKPT_DIR'], CHECKPOINT),
                           map_location=torch.device(DEVICE))
model.load_state_dict(model_weights)
model = model.to(DEVICE)
model.eval()

if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

with torch.no_grad():
    i = 0
    while i < num_samples:
        batch_samples = min(num_samples - i, BATCH_SIZE)

        noise = torch.randn(size=(BATCH_SIZE, GAN_Z_SIZE, 1, 1), device=DEVICE)
        output = model(noise).cpu().detach()
        fake_samples = (output + 1) / 2

        for j in range(batch_samples):
            sample = np.clip(output[j, :, :, :].permute(1, 2, 0).numpy() * 255, a_min=0, a_max=255).astype(np.uint8)
            img = Image.fromarray(sample, mode='RGB')
            img.save(os.path.join('tmp', f'{i * BATCH_SIZE + j}.png'))

        i += batch_samples
