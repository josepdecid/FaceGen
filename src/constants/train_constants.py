import torch

EPOCHS = 100
BATCH_SIZE = 64
Z_SIZE = 300
SHUFFLE_TRAIN = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
