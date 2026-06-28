import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backend.predictor import (
    brain_model,
    pneumonia_model,
    DEVICE
)

# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()

])

# =====================================================
# GradCAM Class
# =====================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.target_layer = target_layer

        self.gradients = None

        self.activations = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, input, output):

        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def remove_hooks(self):

        self.forward_handle.remove()

        self.backward_handle.remove()

    def generate(self, image_tensor):

        self.model.zero_grad()

        output = self.model(image_tensor)

        predicted_class = output.argmax(dim=1)

        score = output[:, predicted_class]

        score.backward()

        gradients = self.gradients[0]

        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(

            activations.shape[1:],

            device=DEVICE

        )

        for i in range(len(weights)):

            cam += weights[i] * activations[i]

        cam = torch.relu(cam)

        cam -= cam.min()

        cam /= cam.max() + 1e-8

        return cam.detach().cpu().numpy()

# =====================================================
# Generate GradCAM
# =====================================================

def generate_gradcam(

    image_path,

    model_name="brain"

):

    image = Image.open(

        image_path

    ).convert("RGB")

    tensor = transform(

        image

    ).unsqueeze(0).to(DEVICE)

    # -----------------------------------------

    if model_name.lower() == "brain":

        model = brain_model

    else:

        model = pneumonia_model

    model.eval()

    # -----------------------------------------
    # EfficientNet Last Feature Block
    # -----------------------------------------

    target_layer = model.model.features[-1]

    gradcam = GradCAM(

        model,

        target_layer

    )

    heatmap = gradcam.generate(

        tensor

    )

    gradcam.remove_hooks()

    # -----------------------------------------
    # Original Image
    # -----------------------------------------

    original = cv2.imread(image_path)

    original = cv2.resize(

        original,

        (224, 224)

    )

    # -----------------------------------------
    # Heatmap
    # -----------------------------------------

    heatmap = cv2.resize(

        heatmap,

        (224, 224)

    )

    heatmap = np.uint8(

        heatmap * 255

    )

    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )

    overlay = cv2.addWeighted(

        original,

        0.60,

        heatmap,

        0.40,

        0

    )

    overlay = cv2.cvtColor(

        overlay,

        cv2.COLOR_BGR2RGB

    )

    return overlay