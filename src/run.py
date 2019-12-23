import os

from dotenv import load_dotenv
from torchvision.transforms import ToTensor

from models.gan.GAN import GAN
from models.trainer import train

from dataset.UTKFaceDataset import UTKFaceDataset

if __name__ == '__main__':
    load_dotenv()
    dataset = UTKFaceDataset(os.environ['DATASET_PATH'], transform=ToTensor())

    model = GAN()
    train(model, dataset)
