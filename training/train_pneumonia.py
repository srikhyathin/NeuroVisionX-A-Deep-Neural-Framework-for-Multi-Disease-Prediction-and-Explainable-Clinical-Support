import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

from models.image_model import BrainTumorNet


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("NeuroVisionX - Pneumonia Detection Training")
print("=" * 60)
print("Device :", device)

# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor()

])

# ==========================================================
# Dataset
# ==========================================================

train_dataset = datasets.ImageFolder(

    "data/pneumonia/train",

    transform=transform

)

test_dataset = datasets.ImageFolder(

    "data/pneumonia/test",

    transform=transform

)

print("\nClasses :", train_dataset.classes)

print("Training Images :", len(train_dataset))

print("Testing Images :", len(test_dataset))

# ==========================================================
# DataLoader
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=True

)

test_loader = DataLoader(

    test_dataset,

    batch_size=16,

    shuffle=False

)

# ==========================================================
# Model
# ==========================================================

model = BrainTumorNet(

    num_classes=2

).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    model.parameters(),

    lr=0.0001

)

# ==========================================================
# Training
# ==========================================================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(

            outputs,

            labels

        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(

            outputs,

            1

        )

        total += labels.size(0)

        correct += (

            predicted == labels

        ).sum().item()

    accuracy = 100 * correct / total

    print(

        f"Epoch {epoch+1}/{epochs}"

        f" | Loss: {running_loss:.4f}"

        f" | Accuracy: {accuracy:.2f}%"

    )

# ==========================================================
# Testing
# ==========================================================

model.eval()

correct = 0

total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(

            outputs,

            1

        )

        total += labels.size(0)

        correct += (

            predicted == labels

        ).sum().item()

test_accuracy = 100 * correct / total

print("\nTest Accuracy :", round(test_accuracy,2), "%")

# ==========================================================
# Save Model
# ==========================================================

os.makedirs(

    "saved_models",

    exist_ok=True

)

torch.save(

    model.state_dict(),

    "saved_models/pneumonia_model.pth"

)

print("\nModel Saved Successfully!")

print("saved_models/pneumonia_model.pth")

print("=" * 60)