import torch
import torch.nn as nn

class NeuroFusion(nn.Module):

    def __init__(self):

        super(NeuroFusion, self).__init__()

        self.fusion = nn.Sequential(

            nn.Linear(44, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 16)

        )

    def forward(self, x):

        return self.fusion(x)