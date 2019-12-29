import os
from glob import glob

import torch
from PIL import Image
from torchvision import get_image_backend
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize


class FaceNoFaceDataset(VisionDataset):
    """
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

    Attributes:
        samples (list): List of sample paths
    """

    def __init__(self, root_positive, root_negative, transform=None):
        super().__init__(root_positive, transform=transform)

        self.positive_samples = FaceNoFaceDataset.__make_dataset(self.root)
        self.negative_samples = FaceNoFaceDataset.__make_dataset(root_negative)

        self.samples = self.positive_samples + self.negative_samples
        self.labels = [1 for _ in range(len(self.positive_samples))] + [0 for _ in range(len(self.negative_samples))]

        if len(self.samples) == 0:
            raise RuntimeError(f'Found 0 .jpg files in {self.root}')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.__loader(path)
        if index >= len(self.positive_samples):
            sample = Resize(size=(200, 200))(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.labels[index]

    def __len__(self):
        return len(self.positive_samples + self.negative_samples)

    @staticmethod
    def __make_dataset(path: str):
        images = []
        path = os.path.expanduser(path)
        print(path)
        for f_path in glob(os.path.join(path, '*.jpg')):
            images.append(f_path)
        for f_path in glob(os.path.join(path, '*.png')):
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
            return FaceNoFaceDataset.__pil_loader(path)

    @staticmethod
    def __loader(path):
        if get_image_backend() == 'accimage':
            return FaceNoFaceDataset.__acc_loader(path)
        else:
            return FaceNoFaceDataset.__pil_loader(path)
