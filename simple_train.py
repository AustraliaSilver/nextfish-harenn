import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Load data
with open("merged_data.jsonl", "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(data)} samples")


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        # Dummy features
        features = torch.zeros(768)
        # Labels
        eval_score = pos.get("eval_score", 0) / 100.0
        tau = pos.get("tau", 0.5)
        rho = pos.get("rho", 0.5)
        rs = pos.get("rs", 0.5)
        labels = torch.tensor([eval_score, tau, rho, rs], dtype=torch.float32)
        return features, labels


# Split
train_data = data[: int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)) :]

train_dataset = SimpleDataset(train_data)
val_dataset = SimpleDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 4)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train
for epoch in range(5):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        pred = model(features)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            pred = model(features)
            loss = criterion(pred, labels)
            val_loss += loss.item()
    print(f"Val Loss: {val_loss / len(val_loader):.4f}")

torch.save(model.state_dict(), "simple_model.pth")
print("Model saved")
