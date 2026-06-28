import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from models.heart_model import HeartNet

df = pd.read_csv(
    "data/heart/heart_processed.csv"
)

X = df.drop("target", axis=1)

y = df["target"]

y = (y > 0).astype(int)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = torch.tensor(
    X,
    dtype=torch.float32
)

y = torch.tensor(
    y.values,
    dtype=torch.long
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
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

model = HeartNet(
    input_size=X.shape[1]
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 30

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "saved_models/heart_model.pth"
)

print(
    "Heart model saved."
)