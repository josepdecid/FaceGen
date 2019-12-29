import torch

EPOCHS = 10
BATCH_SIZE = 32
Z_SIZE = 100
SHUFFLE_TRAIN = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
