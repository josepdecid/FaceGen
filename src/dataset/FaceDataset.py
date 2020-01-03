import os
from glob import glob

import torch
from torchvision import get_image_backend
from torchvision.datasets import VisionDataset
from PIL import Image


class FaceDataset(VisionDataset):
    """
    Args:
        root_positive (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

    Attributes:
        samples (list): List of sample paths
    """

    def __init__(self, root_positive, root_negative, transform=None):
        super().__init__(root_positive, transform=transform)

        self.positive_samples = FaceDataset.__make_dataset(root_positive)
        self.negative_samples = FaceDataset.__make_dataset(root_negative)

        self.samples = self.positive_samples + self.negative_samples
        self.labels = [1.0 for _ in range(len(self.positive_samples))] + \
                      [0.2 for _ in range(len(self.negative_samples))]

        if len(self.samples) == 0:
            raise RuntimeError(f'Found 0 files in {self.root}')

        if len(self.negative_samples):
            print(f'Dataset loaded with {len(self.positive_samples)} positive samples'
                  f'and {len(self.negative_samples)} negative samples')
        else:
            print(f'Dataset loaded with {len(self.samples)} samples')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.__loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def __make_dataset(path: str):
        images = []
        path = os.path.expanduser(path)
        print(path)
        for f_path in glob(os.path.join(path, '*.jpg')):
            images.append(f_path)
        return images

    @staticmethod
    def __pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def __acc_loader(path):
        try:
            # TODO: Install ACCImage (Only available Conda-Forge)
            import accimage
            return accimage.Image(path)
        except IOError:
            return FaceDataset.__pil_loader(path)

    @staticmethod
    def __loader(path):
        if get_image_backend() == 'accimage':
            return FaceDataset.__acc_loader(path)
        else:
            return FaceDataset.__pil_loader(path)
