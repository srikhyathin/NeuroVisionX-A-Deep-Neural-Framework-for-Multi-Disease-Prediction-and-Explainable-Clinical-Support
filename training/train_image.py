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


class ImageTrainer:

    def __init__(self,
                 dataset_path,
                 save_path,
                 num_classes):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dataset_path = dataset_path

        self.save_path = save_path

        self.transform = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(10),

            transforms.ToTensor()

        ])

        self.dataset = datasets.ImageFolder(
            dataset_path,
            transform=self.transform
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=16,
            shuffle=True
        )

        self.model = BrainTumorNet(
            num_classes=num_classes
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001
        )

    def train(self, epochs=10):

        print("=" * 50)
        print("NeuroVisionX Image Trainer")
        print("=" * 50)

        for epoch in range(epochs):

            self.model.train()

            running_loss = 0

            correct = 0

            total = 0

            for images, labels in self.loader:

                images = images.to(self.device)

                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                loss.backward()

                self.optimizer.step()

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

        torch.save(

            self.model.state_dict(),

            self.save_path

        )

        print("\nModel Saved Successfully!")

        print(self.save_path)


# ------------------------------------------------------
# Main
# ------------------------------------------------------

if __name__ == "__main__":

    trainer = ImageTrainer(

        dataset_path="data/brain_tumor/Training",

        save_path="saved_models/brain_tumor_model.pth",

        num_classes=4

    )

    trainer.train(epochs=5)