import os
from glob import glob
from torchvision import get_image_backend
from torchvision.datasets import VisionDataset
from PIL import Image


class UTKFaceDataset(VisionDataset):
    """
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

    Attributes:
        samples (list): List of sample paths
    """

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

        self.samples = UTKFaceDataset.__make_dataset(self.root)
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
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

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
            return UTKFaceDataset.__pil_loader(path)

    @staticmethod
    def __loader(path):
        if get_image_backend() == 'accimage':
            return UTKFaceDataset.__acc_loader(path)
        else:
            return UTKFaceDataset.__pil_loader(path)
