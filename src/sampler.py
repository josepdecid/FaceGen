import torch

from utils.train_constants import DEVICE, Z_SIZE
from models.gan.Generator import Generator


def generate_samples(model_path, num_samples):
    model_weights = torch.load(model_path)

    G: Generator = Generator()
    G = G.to(DEVICE)

    G.load_state_dict(model_weights)
    G.eval()

    samples = []
    for i in range(num_samples):
        noise = torch.randn(1, Z_SIZE, 1, 1, device=DEVICE)
        img = G(noise)
        samples.append(img)
