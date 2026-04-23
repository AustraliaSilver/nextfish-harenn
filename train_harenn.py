#!/usr/bin/env python3
"""
HARENN Training Script - Full Version

    Trains a neural network to predict:
1. Evaluation score (centipawns)
2. Tactical complexity (tau)
3. Game result
4. Horizon risk (rho)
5. Resolution score (rs)
6. Move criticality map (MCS 64x64)

Usage:
    python train_harenn.py --data ./data --epochs 10
"""

import argparse
import json
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


def fen_to_features(fen: str) -> np.ndarray:
    """
    Convert FEN to NNUE-style feature vector.
    Input: FEN string
    Output: Feature vector (768 features)

    Features: 64 squares × 12 piece types
    Piece types: P, N, B, R, Q, K (white), p, n, b, r, q, k (black)
    """
    parts = fen.split()
    board = parts[0]
    stm = 0 if parts[1] == "w" else 1  # 0=white to move

    # 64 squares × 12 pieces = 768 features
    features = np.zeros(768, dtype=np.float32)

    piece_map = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    square = 0
    for char in board:
        if char.isdigit():
            square += int(char)
        elif char != "/":
            if char in piece_map and square < 64:
                features[square * 12 + piece_map[char]] = 1.0
            square += 1

    return features, stm


class HARENNDataset(Dataset):
    """Dataset for HARENN training"""

    def __init__(self, data_dir: str, max_samples: int = None):
        self.samples = []
        self.load_data(data_dir, max_samples)

        # Compute normalization stats (only if eval_score exists)
        eval_scores = [s["eval_score"] for s in self.samples if "eval_score" in s and s["eval_score"] is not None]
        self.has_eval = len(eval_scores) > 0
        if self.has_eval:
            self.eval_mean = np.mean(eval_scores)
            self.eval_std = np.std(eval_scores) + 1e-8
        else:
            self.eval_mean = 0.0
            self.eval_std = 1.0

    def load_data(self, data_dir: str, max_samples: int = None):
        """Load all JSON files"""
        files = glob.glob(os.path.join(data_dir, "*.json"))

        for f in files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    for pos in data.get("positions", []):
                        # Require full labels
                        if "mcs_map" not in pos or "rho" not in pos or "rs" not in pos:
                            continue
                        self.samples.append(pos)
                        if max_samples and len(self.samples) >= max_samples:
                            break
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert FEN to features
        features, stm = fen_to_features(sample["fen"])

        # Normalize eval
        eval_score = sample.get("eval_score", None)
        if eval_score is None:
            eval_normalized = 0.0
            eval_mask = 0.0
        else:
            eval_normalized = (eval_score - self.eval_mean) / self.eval_std
            eval_mask = 1.0

        tau = float(sample["tau"])
        game_result_raw = float(sample.get("game_result", 1))
        # Map game_result from {0,1,2} -> {0.0,0.5,1.0}
        game_result = game_result_raw / 2.0
        rho = float(sample["rho"])
        rs = float(sample["rs"])

        mcs_map = np.array(sample["mcs_map"], dtype=np.float32)
        if mcs_map.size != 4096:
            raise ValueError("mcs_map must have 4096 values")
        mcs_map = mcs_map.reshape(64, 64)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "stm": torch.tensor(stm, dtype=torch.float32),
            "eval": torch.tensor(eval_normalized, dtype=torch.float32),
            "eval_mask": torch.tensor(eval_mask, dtype=torch.float32),
            "tau": torch.tensor(tau, dtype=torch.float32),
            "result": torch.tensor(game_result, dtype=torch.float32),
            "rho": torch.tensor(rho, dtype=torch.float32),
            "rs": torch.tensor(rs, dtype=torch.float32),
            "mcs": torch.tensor(mcs_map, dtype=torch.float32),
        }


class HARENNNet(nn.Module):
    """
    HARENN Network - Hybrid Architecture

    Input: 768 (features) + 1 (stm) = 769
    Output: eval, tau, result, rho, rs, mcs(64x64)
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()

        # Input layer
        self.input = nn.Linear(769, hidden_size)

        # Hidden layers
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size // 2)

        # Output heads
        self.eval_head = nn.Linear(hidden_size // 2, 1)
        self.tau_head = nn.Linear(hidden_size // 2, 1)
        self.result_head = nn.Linear(hidden_size // 2, 1)
        self.horizon_head = nn.Linear(hidden_size // 2, 1)
        self.resolution_head = nn.Linear(hidden_size // 2, 1)

        # MCS head: 256 -> 64 -> 4096
        self.mcs_h1 = nn.Linear(hidden_size // 2, 64)
        self.mcs_out = nn.Linear(64, 4096)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x, stm):
        # Concatenate features with side to move
        x = torch.cat([x, stm.unsqueeze(1)], dim=1)

        # Forward pass
        x = self.relu(self.bn1(self.input(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.hidden1(x)))
        x = self.dropout(x)
        x = self.relu(self.hidden2(x))

        # Outputs
        eval_out = self.eval_head(x)
        tau_out = torch.sigmoid(self.tau_head(x))
        result_out = torch.sigmoid(self.result_head(x))
        horizon_out = torch.sigmoid(self.horizon_head(x))
        resolution_out = torch.sigmoid(self.resolution_head(x))

        mcs = self.relu(self.mcs_h1(x))
        mcs_out = torch.sigmoid(self.mcs_out(mcs))
        mcs_out = mcs_out.view(-1, 64, 64)

        return eval_out, tau_out, result_out, horizon_out, resolution_out, mcs_out


def train_epoch(model, dataloader, optimizer, criterion, device, eval_scaler):
    """Train one epoch"""
    model.train()
    total_loss = 0
    eval_losses = []
    tau_losses = []
    result_losses = []
    horizon_losses = []
    resolution_losses = []
    mcs_losses = []

    for batch in dataloader:
        features = batch["features"].to(device)
        stm = batch["stm"].to(device)
        eval_target = batch["eval"].to(device)
        eval_mask = batch["eval_mask"].to(device)
        tau_target = batch["tau"].to(device)
        result_target = batch["result"].to(device)
        rho_target = batch["rho"].to(device)
        rs_target = batch["rs"].to(device)
        mcs_target = batch["mcs"].to(device)

        optimizer.zero_grad()

        # Forward
        eval_out, tau_out, result_out, rho_out, rs_out, mcs_out = model(features, stm)

        # Losses
        if eval_mask.sum() > 0:
            per_eval = torch.nn.functional.smooth_l1_loss(
                eval_out.squeeze(), eval_target.squeeze(), reduction="none"
            )
            eval_loss = (per_eval * eval_mask).sum() / eval_mask.sum()
        else:
            eval_loss = torch.tensor(0.0, device=device)
        tau_loss = criterion(tau_out.squeeze(), tau_target)
        result_loss = criterion(result_out.squeeze(), result_target)
        horizon_loss = criterion(rho_out.squeeze(), rho_target)
        resolution_loss = criterion(rs_out.squeeze(), rs_target)
        mcs_loss = criterion(mcs_out, mcs_target)

        # Weighted multi-task loss
        loss = (
            eval_loss * 1.0
            + tau_loss * 1.0
            + result_loss * 0.2
            + horizon_loss * 0.6
            + resolution_loss * 0.6
            + mcs_loss * 0.3
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        eval_losses.append(eval_loss.item())
        tau_losses.append(tau_loss.item())
        result_losses.append(result_loss.item())
        horizon_losses.append(horizon_loss.item())
        resolution_losses.append(resolution_loss.item())
        mcs_losses.append(mcs_loss.item())

    return {
        "total": total_loss / len(dataloader),
        "eval": np.mean(eval_losses),
        "tau": np.mean(tau_losses),
        "result": np.mean(result_losses),
        "horizon": np.mean(horizon_losses),
        "resolution": np.mean(resolution_losses),
        "mcs": np.mean(mcs_losses),
    }


def validate(model, dataloader, criterion, device, eval_scaler):
    """Validate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            stm = batch["stm"].to(device)
            eval_target = batch["eval"].to(device)
            eval_mask = batch["eval_mask"].to(device)
            tau_target = batch["tau"].to(device)
            result_target = batch["result"].to(device)
            rho_target = batch["rho"].to(device)
            rs_target = batch["rs"].to(device)
            mcs_target = batch["mcs"].to(device)

            eval_out, tau_out, result_out, rho_out, rs_out, mcs_out = model(features, stm)

            if eval_mask.sum() > 0:
                per_eval = torch.nn.functional.smooth_l1_loss(
                    eval_out.squeeze(), eval_target.squeeze(), reduction="none"
                )
                eval_loss = (per_eval * eval_mask).sum() / eval_mask.sum()
            else:
                eval_loss = torch.tensor(0.0, device=device)
            tau_loss = criterion(tau_out.squeeze(), tau_target)
            result_loss = criterion(result_out.squeeze(), result_target)
            horizon_loss = criterion(rho_out.squeeze(), rho_target)
            resolution_loss = criterion(rs_out.squeeze(), rs_target)
            mcs_loss = criterion(mcs_out, mcs_target)

            loss = (
                eval_loss * 1.0
                + tau_loss * 1.0
                + result_loss * 0.2
                + horizon_loss * 0.6
                + resolution_loss * 0.6
                + mcs_loss * 0.3
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="HARENN Training")
    parser.add_argument("--data", "-d", default="./data", help="Data directory")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Epochs")
    parser.add_argument("--batch", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size")
    parser.add_argument("--save", "-s", default="./harenn_model.pth", help="Save model")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to load"
    )
    parser.add_argument(
        "--resume", "-r", action="store_true", help="Resume from saved model"
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading data from: {args.data}")
    dataset = HARENNDataset(args.data, args.max_samples)

    # Split train/val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=0
    )

    # Eval scaler for denormalization
    eval_scaler = {"mean": dataset.eval_mean, "std": dataset.eval_std}

    # Model
    print(f"Creating model with hidden={args.hidden}...")
    model = HARENNNet(hidden_size=args.hidden).to(device)

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume and os.path.exists(args.save):
        print(f"Resuming from {args.save}...")
        checkpoint = torch.load(args.save, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        # Use saved scaler
        if "eval_mean" in checkpoint:
            eval_scaler = {
                "mean": checkpoint["eval_mean"],
                "std": checkpoint["eval_std"],
            }
            dataset.eval_mean = checkpoint["eval_mean"]
            dataset.eval_std = checkpoint["eval_std"]
        print("Model loaded!")

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    # Training loop
    best_val_loss = float("inf")

    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, eval_scaler
        )
        val_loss = validate(model, val_loader, criterion, device, eval_scaler)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:2d}/{args.epochs} | "
            f"Train Loss: {train_loss['total']:.4f} (eval:{train_loss['eval']:.4f}, tau:{train_loss['tau']:.4f}, "
            f"rho:{train_loss['horizon']:.4f}, rs:{train_loss['resolution']:.4f}, mcs:{train_loss['mcs']:.4f}) | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "eval_mean": dataset.eval_mean,
                    "eval_std": dataset.eval_std,
                    "args": args,
                },
                args.save,
            )

    print(f"\nBest model saved to: {args.save}")

    # Test inference
    print("\nTesting inference...")
    checkpoint = torch.load(args.save, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        sample = val_dataset[0]
        features = sample["features"].unsqueeze(0).to(device)
        stm = sample["stm"].unsqueeze(0).to(device)

        eval_out, tau_out, result_out, rho_out, rs_out, mcs_out = model(features, stm)

        # Denormalize eval
        eval_pred = eval_out.item() * dataset.eval_std + dataset.eval_mean

        print(f"Sample prediction:")
        print(
            f"  Eval: {eval_pred:.1f} (target: {sample['eval'].item() * dataset.eval_std + dataset.eval_mean:.1f})"
        )
        print(f"  Tau: {tau_out.item():.3f} (target: {sample['tau'].item():.3f})")
        print(
            f"  Result: {result_out.item():.3f} (target: {sample['result'].item():.3f})"
        )
        print(f"  Rho: {rho_out.item():.3f} (target: {sample['rho'].item():.3f})")
        print(f"  Rs: {rs_out.item():.3f} (target: {sample['rs'].item():.3f})")


if __name__ == "__main__":
    main()
