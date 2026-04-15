#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import argparse

def fen_to_features(fen: str) -> np.ndarray:
    parts = fen.split()
    board = parts[0]
    features = np.zeros(768, dtype=np.float32)
    piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    square = 0
    for char in board:
        if char.isdigit():
            square += int(char)
        elif char != "/":
            if char in piece_map and square < 64:
                features[square * 12 + piece_map[char]] = 1.0
            square += 1
    return features

class BatchDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        data_path = Path(data_dir)
        print(f"Loading data from {data_path}...")
        
        # Recursive search for JSON files in data_batches
        for json_file in data_path.rglob("*.json"):
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                    if "positions" in data:
                        self.samples.extend(data["positions"])
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        # Also check for .jsonl files (like merged_data.jsonl)
        for jsonl_file in data_path.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} total positions.")
        
        # Compute eval stats for normalization
        evals = [s.get("eval_score", 0) for s in self.samples]
        self.eval_mean = np.mean(evals)
        self.eval_std = np.std(evals) + 1e-8

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = fen_to_features(sample["fen"])
        
        eval_score = sample.get("eval_score", 0)
        # Normalize eval for training stability
        eval_label = (eval_score - self.eval_mean) / self.eval_std
        tau_label = sample.get("tau", 0.5)
        
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "eval": torch.tensor([eval_label], dtype=torch.float32),
            "tau": torch.tensor([tau_label], dtype=torch.float32)
        }

class NextfishModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.eval_head = nn.Linear(128, 1)
        self.tau_head = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        eval_out = self.eval_head(x)
        tau_out = torch.sigmoid(self.tau_head(x))
        return eval_out, tau_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data_batches")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save", default="nextfish_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BatchDataset(args.data)
    if len(dataset) == 0:
        print("No data found!")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = NextfishModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion_eval = nn.MSELoss()
    criterion_tau = nn.BCELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y_eval = batch["eval"].to(device)
            y_tau = batch["tau"].to(device)
            
            optimizer.zero_grad()
            pred_eval, pred_tau = model(x)
            
            loss_eval = criterion_eval(pred_eval, y_eval)
            loss_tau = criterion_tau(pred_tau, y_tau)
            loss = loss_eval + loss_tau
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y_eval = batch["eval"].to(device)
                y_tau = batch["tau"].to(device)
                pred_eval, pred_tau = model(x)
                val_loss += (criterion_eval(pred_eval, y_eval) + criterion_tau(pred_tau, y_tau)).item()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "eval_mean": dataset.eval_mean,
        "eval_std": dataset.eval_std
    }, args.save)
    print(f"Model saved to {args.save}")

if __name__ == "__main__":
    main()
