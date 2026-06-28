import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, input, output):

        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx=None):

        self.model.eval()

        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients

        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)

        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))

        cam = cam - np.min(cam)

        cam = cam / (np.max(cam) + 1e-8)

        return cam


def overlay_heatmap(image_path, cam):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (224, 224))

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay


def preprocess_image(image_path):

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor()

    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    return image