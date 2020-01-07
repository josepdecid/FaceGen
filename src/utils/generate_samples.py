import os

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv

from models.gan.Generator import Generator
from utils.train_constants import DEVICE, GAN_Z_SIZE

load_dotenv()

model: Generator = Generator()
model_weights = torch.load(os.path.join(os.environ['CKPT_DIR'], '96_G.pt'),
                           map_location=torch.device(DEVICE))
model.load_state_dict(model_weights)
model = model.to(DEVICE)
model.eval()

i = 0
samples = []
with torch.no_grad():
    while i < 8:
        noise = torch.randn(size=(1, GAN_Z_SIZE, 1, 1), device=DEVICE)
        output = model(noise).squeeze()
        fake_samples = (output + 1) / 2

        plt.imshow(fake_samples.permute(1, 2, 0).cpu())
        plt.show()

        keep = input('Keep? ')
        if keep == 'y':
            samples.append(fake_samples.permute(1, 2, 0).cpu())
            i += 1

f, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 4), gridspec_kw={'wspace': 0, 'hspace': 0})

for i, sample in enumerate(samples):
    ax[divmod(i, 4)].imshow(sample)
    ax[divmod(i, 4)].axis('off')

plt.show()
