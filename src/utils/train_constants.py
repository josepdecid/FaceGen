import torch

EPOCHS = 1000
BATCH_SIZE = 64

GAN_Z_SIZE = 100
VAE_Z_SIZE = 150

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 64
