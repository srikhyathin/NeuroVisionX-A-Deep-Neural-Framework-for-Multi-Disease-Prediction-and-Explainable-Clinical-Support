import torch
import torch.nn as nn

class DiabetesNet(nn.Module):

    def __init__(self):
        super(DiabetesNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)