import argparse
import os
import random
from datetime import datetime

import torch
from dotenv import load_dotenv
from torchvision import transforms

from utils.train_constants import GA_IMG_SIZE
from dataset.FaceDataset import FaceDataset
from models.autoencoder.vae import VAE
from models.evolutionary.face_classifier import FaceClassifier
from models.evolutionary.genetic_algorithm import GeneticAlgorithm
from models.gan.Discriminator import Discriminator
from models.gan.Generator import Generator
from sampler import generate_samples
from trainers.ga_trainer import GATrainer
from trainers.gan_trainer import GANTrainer
from trainers.vae_trainer import VAETrainer


def main(args):
    if args.generate is None:
        load_dotenv()

        # Set random seed for reproducibility
        manual_seed = random.randint(1, 1e10) if args.seed is None else args.seed
        print(f'Random Seed: {manual_seed}')

        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        log_tag = ('-'.join(str(datetime.now()).split()) + f'_{manual_seed}').replace(':', '-')
        if not os.path.exists(os.environ['CKPT_DIR']):
            os.mkdir(os.environ['CKPT_DIR'])

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(distortion_scale=0.2),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(degrees=10),
            # transforms.Resize(size=(200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # dataset = FaceDataset(os.environ['DATASET_PATH'], transform=transform)

        ################################
        # Using Genetic Algorithms (GAs)
        ################################

        if args.model == 'GA':
            # Create a model for classifying between Faces or Non-Faces.
            # We use the output of this model as the fitness function of our Genetic Algorithm.
            model: FaceClassifier = FaceClassifier()

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.Resize(size=(GA_IMG_SIZE, GA_IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            dataset = FaceDataset(root_positive=os.environ['DATASET_PATH'],
                                  root_negative=os.environ['CIFAR_PATH'],
                                  transform=transform)

            if args.pretrained is None:
                # Train model from scratch using our main dataset as positive samples and CIFAR10 as negatives.
                # fnf_dataset = FaceNoFaceDataset(root_positive=os.path.join(os.environ['FNF_DATASET_PATH'], 'Positive'),
                #                                root_negative=os.path.join(os.environ['FNF_DATASET_PATH'], 'Negative'),
                #                                transform=transform)
                model = FaceClassifier()
                trainer = GATrainer(model, dataset, log_tag=log_tag)
                trainer.train()
            else:
                # Load pretrained model from the checkpoints directory.
                path = os.path.join(os.environ['CKPT_DIR'], f'{args.pretrained}.pt')
                model_weights = torch.load(path)
                model.load_state_dict(model_weights)

            # Run Genetic Algorithm
            model.eval()
            with torch.no_grad():
                GA = GeneticAlgorithm(model, par=False, log_tag=log_tag)
                GA.run(dataset)

        ##############################################
        # Using Generative Adversarial Networks (GANs)
        ##############################################

        elif args.model == 'GAN':
            G = Generator()
            D = Discriminator()
            trainer = GANTrainer(G=G, D=D, dataset=dataset, log_tag=log_tag)
            trainer.train()

        ########################################
        # Using Variational Auto Encoders (VAEs)
        ########################################

        else:
            model = VAE()
            trainer = VAETrainer(model=model, dataset=dataset, log_tag=log_tag)
            trainer.train()
    else:
        generate_samples(model_path=args.generate[0],
                         num_samples=args.generate[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN for Face Generation.')
    parser.add_argument('model', type=str, choices=['GA', 'VAE', 'GAN'],
                        help='Choose model between VAE and GAN')
    parser.add_argument('--pretrained', type=str, required=False,
                        help='')
    parser.add_argument('--generate', nargs=2, required=False,
                        help='Whether to generate a sample instead of training the model. '
                             'Need to specify the model file name located in folder `/checkpoints`. '
                             'You also need to specify the number of samples to generate')
    parser.add_argument('--seed', type=int, required=False,
                        help='Set manual random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Train the model in a CUDA GPU (Default to CPU if no available)')

    print(parser.parse_args())
    main(parser.parse_args())
