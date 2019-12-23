import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from dataset.UTKFaceDataset import UTKFaceDataset

if __name__ == '__main__':
    load_dotenv()
    dataset = UTKFaceDataset(os.environ['DATASET_PATH'])

    plt.imshow(dataset[0])
    plt.show()
