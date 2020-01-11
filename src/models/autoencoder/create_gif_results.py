import os

import torch
import torchvision.utils as vutils
from PIL import Image

from models.autoencoder.vae import VAE
from utils.train_constants import DEVICE, VAE_Z_SIZE

base_path = 'D:\\USUARIS\\gonzalorecio\\CI\\checkpoints\\VAE_2020-01-10-18-01-48.713216_8421672063'

model = VAE()
model = model.to(DEVICE)
model.eval()

images = []

fixed_noise = torch.randn(size=(1, GAN_Z_SIZE, 1, 1), device=DEVICE)
for i in range(48):
    checkpoint_path = os.path.join(base_path, f'{i}.pt')
    model_weights = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
    model.load_state_dict(model_weights)

    fake_samples = model.decode(fixed_latent)
    fake_samples = 255 * (fake_samples.cpu() + 1) / 2
    fake_samples = vutils.make_grid(fake_samples, padding=1, nrow=8)

    images.append(fake_samples.detach().squeeze().permute(1, 2, 0).numpy().astype('uint8'))

img, *images = [Image.fromarray(image) for image in images]
img.save(fp='generations.gif', format='GIF', append_images=images,
         save_all=True, duration=150, loop=0)
