import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import torch

from models.neurofusion import NeuroFusion

model = NeuroFusion()

dummy_input = torch.randn(
    10,
    44
)

output = model(dummy_input)

print("Fusion Output Shape:")

print(output.shape)

torch.save(
    model.state_dict(),
    "saved_models/neurofusion.pth"
)

print("NeuroFusion saved.")