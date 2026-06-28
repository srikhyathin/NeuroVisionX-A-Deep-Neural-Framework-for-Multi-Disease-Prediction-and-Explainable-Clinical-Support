import torch.nn as nn
from torchvision.models import efficientnet_b0

class BrainTumorNet(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.model = efficientnet_b0(weights="DEFAULT")

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)