import torch

EPOCHS = 1000
BATCH_SIZE = 64
Z_SIZE = 150
SHUFFLE_TRAIN = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GA_IMG_SIZE = 100
