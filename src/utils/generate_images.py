import os
import shutil

import numpy as np
import torch
from PIL import Image

from models.gan.Generator import Generator
from utils.train_constants import DEVICE, BATCH_SIZE, GAN_Z_SIZE, VAE_Z_SIZE


def generate_images(checkpoint, samples, ModelClass):
    model = ModelClass()
    model_weights = torch.load(os.path.join(os.environ['CKPT_DIR'], checkpoint) + '.pt',
                               map_location=torch.device(DEVICE))
    model.load_state_dict(model_weights)
    model = model.to(DEVICE)
    model.eval()

    if os.path.exists('generated_faces'):
        shutil.rmtree('generated_faces')
    os.mkdir('generated_faces')

    with torch.no_grad():
        i = 0
        while i < samples:
            batch_samples = min(samples - i, BATCH_SIZE)

            if ModelClass == Generator:
                noise = torch.randn(size=(samples, GAN_Z_SIZE, 1, 1), device=DEVICE)
                output = model(noise).cpu().detach()
            else:
                latent = torch.randn(size=(samples, VAE_Z_SIZE), device=DEVICE)
                output = model.decode(latent).cpu().detach()
            fake_samples = (output + 1) / 2

            for j in range(batch_samples):
                sample = np.clip(fake_samples[j, :, :, :].permute(1, 2, 0).numpy() * 255, a_min=0, a_max=255)
                img = Image.fromarray(sample.astype(np.uint8), mode='RGB')
                img.save(os.path.join('generated_faces', f'{i * BATCH_SIZE + j}.png'))

            i += batch_samples
