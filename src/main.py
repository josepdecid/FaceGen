import argparse
import os
import random

import torch
from dotenv import load_dotenv
from torchvision import transforms

from dataset.UTKFaceDataset import UTKFaceDataset
from models.gan.Discriminator import Discriminator
from models.gan.Generator import Generator
from models.sampler import generate_samples
from models.trainer import Trainer


def main(args):
    if args.generate is None:
        load_dotenv()

        # Set random seed for reproducibility
        manual_seed = random.randint(1, 1e10) if args.seed is None else args.seed
        print(f'Random Seed: {manual_seed}')

        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = UTKFaceDataset(os.environ['DATASET_PATH'], transform=transform)

        G = Generator()
        D = Discriminator()

        trainer = Trainer(G=G, D=D, dataset=dataset)
        trainer.train()
    else:
        generate_samples(model_path=args.generate[0],
                         num_samples=args.generate[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN for Face Generation.')
    parser.add_argument('--generate', nargs=2, required=False,
                        help='Whether to generate a sample instead of training the model.'
                             'Need to specify the model file name located in folder `/checkpoints`.'
                             'Can also specify the number of samples to generate')
    parser.add_argument('--seed', type=int, required=False,
                        help='Set manual random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Train the model in a CUDA GPU (Default to CPU if no available)')

    print(parser.parse_args())
    main(parser.parse_args())
