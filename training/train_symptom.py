import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models.symptom_model import SymptomNet

# ----------------------------------------------------
# Load Processed Dataset
# ----------------------------------------------------

df = pd.read_csv(
    "data/symptoms/processed_symptoms.csv"
)

X = df.drop(
    "Disease",
    axis=1
).values

y = df["Disease"].values

print("Number of Features:", X.shape[1])
print("Number of Diseases:", len(set(y)))

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y,
    dtype=torch.long
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# ----------------------------------------------------
# Model
# ----------------------------------------------------

model = SymptomNet(
    input_size=X.shape[1],
    num_classes=len(torch.unique(y))
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 50

# ----------------------------------------------------
# Training
# ----------------------------------------------------

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

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
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss={total_loss:.4f} | "
        f"Accuracy={accuracy:.2f}%"
    )

# ----------------------------------------------------
# Save Model
# ----------------------------------------------------

torch.save(
    model.state_dict(),
    "saved_models/symptom_model.pth"
)

print("\nTraining Completed.")
print("Model Saved Successfully.")