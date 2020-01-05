import os

import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import matplotlib.pyplot as plt

from src.models.evolutionary.face_classifier import FaceClassifier
from utils.train_constants import GA_IMG_SIZE

# img = Image.open('lucy.jpg')
# img = Image.open('truck.jpg')
# img = Image.open('face2.jpg')
img = Image.open('faceGA2.jpg')
i = plt.imread('faceGA2.jpg')
print(i.min(), i.max(), i.shape)

transforms = Compose([
    Resize(size=(GA_IMG_SIZE, GA_IMG_SIZE)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

img = transforms(img.float())
img = img.view(1, *img.size())

print(img.max(), img.min())

model = FaceClassifier()

path = os.path.join('..\\..\\..\\checkpoints\\GA_2020-01-04-13-06-32.860282_3787031108\\4.pt')
model_weights = torch.load(path)
model.load_state_dict(model_weights)
model.eval()

print(model(img))
