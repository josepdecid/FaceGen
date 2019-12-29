import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class FaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(pretrained=True, progress=True)

        # Freeze all layers
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x.view(-1)
