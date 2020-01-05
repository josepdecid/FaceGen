import os
from glob import glob

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from torchvision import transforms

from dataset.FaceDataset import FaceDataset
from models.autoencoder.vae import VAE
from utils.train_constants import DEVICE, VAE_Z_SIZE, IMG_SIZE

SAMPLES = 5

if __name__ == '__main__':
    load_dotenv()
    model: VAE = VAE()
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = FaceDataset(os.environ['DATASET_PATH'], transform=transform)
    real_samples = torch.stack([dataset[i][0] for i in range(SAMPLES)])

    models_path = sorted(glob(os.path.join(os.environ['CKPT_DIR'], 'VAE', '*.pt')))

    f, ax = plt.subplots(nrows=len(models_path) + 1, ncols=2 * SAMPLES,
                         figsize=(4 * SAMPLES, 2 * len(models_path)),
                         gridspec_kw={'wspace': 0, 'hspace': 0})
    # plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    latent = torch.randn(size=(SAMPLES, VAE_Z_SIZE)).to(DEVICE)

    real_samples_disp = (255 * ((real_samples + 1) / 2)).permute(0, 2, 3, 1).numpy()
    for j in range(SAMPLES):
        ax[0, j].imshow(real_samples_disp[j, :, :, :].astype('uint8'))
        ax[0, j].axis('off')
        ax[0, j + SAMPLES].axis('off')

    for i, model_path in enumerate(models_path, start=1):
        model_weights = torch.load(model_path, map_location=torch.device(DEVICE))
        model.load_state_dict(model_weights)

        with torch.no_grad():
            reconstructed_images = model(real_samples.to(DEVICE))[0].cpu().detach()
            reconstructed_images = 255 * ((reconstructed_images + 1) / 2)
            reconstructed_images = reconstructed_images.permute(0, 2, 3, 1).numpy()

            for j in range(SAMPLES):
                ax[i, j].imshow(reconstructed_images[j, :, :, :].astype('uint8'))
                ax[i, j].axis('off')

            fake_images = model.decode(latent).cpu().detach()
            fake_images = 255 * ((fake_images + 1) / 2)
            fake_images = fake_images.permute(0, 2, 3, 1).numpy()

            for j in range(SAMPLES):
                ax[i, j + SAMPLES].imshow(fake_images[j, :, :, :].astype('uint8'))
                ax[i, j + SAMPLES].axis('off')
    plt.savefig(f'vae_plot2.png')
    plt.show()
