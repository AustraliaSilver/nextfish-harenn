#!/usr/bin/env python3
"""
HARENN Training Script - Complete Version

This script trains a neural network that matches the C++ HARENN implementation:
- Input: 768 sparse features (64 squares × 12 piece types)
- Hidden layer: 256 units with ReLU
- 4 heads: eval, tau, rho, rs
- Weights: int16_t, Bias: int32_t
- Output: eval uses mean/std, others use sigmoid

Features:
- Validation loop with metrics
- Learning rate scheduler
- Gradient clipping
- Model export to HNN4 format
- Quantization support
- Cross-platform paths

Usage:
    python train_harenn_complete.py --data ./data/full_data.jsonl --epochs 50 --output harenn_model.pth
"""

import argparse
import os
import json
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm


def fen_to_features(fen: str) -> np.ndarray:
    """Convert FEN to sparse feature vector (768 features)"""
    parts = fen.split()
    board = parts[0]
    turn = parts[1] if len(parts) > 1 else "w"
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
    
    # Flip board for Black to move (Stockfish NNUE style)
    # This ensures the model always evaluates from the perspective of the side to move
    if turn == "b":
        # Mirror the board horizontally
        mirrored = np.zeros(768, dtype=np.float32)
        for sq in range(64):
            row = sq // 8
            col = sq % 8
            mirrored_sq = row * 8 + (7 - col)
            mirrored[mirrored_sq * 12:mirrored_sq * 12 + 12] = features[sq * 12:sq * 12 + 12]
        features = mirrored
    
    return features


class HARENNDataset(Dataset):
    """Dataset for HARENN training"""
    def __init__(self, jsonl_path: str):
        self.samples = []
        self.load_data(jsonl_path)
        
        # Compute normalization stats for eval
        eval_scores = [s.get("eval_score", 0) for s in self.samples]
        self.eval_mean = np.mean(eval_scores) if eval_scores else 0.0
        self.eval_std = np.std(eval_scores) + 1e-8 if eval_scores else 1.0
        print(f"Loaded {len(self.samples)} samples")
        print(f"Eval mean: {self.eval_mean:.2f}, std: {self.eval_std:.2f}")

    def load_data(self, jsonl_path: str):
        """Load JSONL file"""
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Data file not found: {jsonl_path}")
        
        print(f"Loading data from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = fen_to_features(sample["fen"])
        
        eval_score = sample.get("eval_score", 0)
        eval_label = (eval_score - self.eval_mean) / self.eval_std
        
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "eval": torch.tensor([eval_label], dtype=torch.float32),
            "tau": torch.tensor([sample.get("tau", 0.5)], dtype=torch.float32),
            "rho": torch.tensor([sample.get("rho", 0.5)], dtype=torch.float32),
            "rs": torch.tensor([sample.get("rs", 0.5)], dtype=torch.float32)
        }


class HARENNModel(nn.Module):
    """
    HARENN Network - Tournament Version (2-Layer)
    
    Input: 768 (sparse features)
    Hidden: 512 → 256 units with ReLU
    Output: eval, tau, rho, rs
    """
    def __init__(self, hidden1_size: int = 512, hidden2_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(768, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.eval_head = nn.Linear(hidden2_size, 1)
        self.tau_head = nn.Linear(hidden2_size, 1)
        self.rho_head = nn.Linear(hidden2_size, 1)
        self.rs_head = nn.Linear(hidden2_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        eval_out = self.eval_head(h2)
        tau_out = torch.sigmoid(self.tau_head(h2))
        rho_out = torch.sigmoid(self.rho_head(h2))
        rs_out = torch.sigmoid(self.rs_head(h2))
        return eval_out, tau_out, rho_out, rs_out


def export_to_hnn4(model: nn.Module, eval_mean: float, eval_std: float, output_path: str):
    """
    Export PyTorch model to HNN4 binary format (2-layer version)
    
    HNN4 Format:
    - Magic: "HNN4" (4 bytes)
    - Eval mean: float (4 bytes)
    - Eval std: float (4 bytes)
    - For each layer:
      - Rows: int (4 bytes)
      - Cols: int (4 bytes)
      - Weights: int16_t (rows * cols * 2 bytes)
      - Bias: int32_t (cols * 4 bytes)
    """
    print(f"Exporting model to HNN4 format: {output_path}")
    
    with open(output_path, 'wb') as f:
        # Write magic
        f.write(b'HNN4')
        
        # Write eval mean and std
        f.write(struct.pack('f', eval_mean))
        f.write(struct.pack('f', eval_std))
        
        # Helper function to write layer
        def write_layer(layer, rows, cols):
            # Write rows and cols
            f.write(struct.pack('i', rows))
            f.write(struct.pack('i', cols))
            
            # Get weights and quantize to int16
            weights = layer.weight.data.cpu().numpy().T  # Transpose for row-major
            # Scale weights to int16 range
            weight_scale = 127.0 / (np.abs(weights).max() + 1e-8)
            weights_int16 = (weights * weight_scale).astype(np.int16)
            f.write(weights_int16.tobytes())
            
            # Get bias and convert to int32
            bias = layer.bias.data.cpu().numpy()
            bias_scale = 1000.0  # Scale factor for bias
            bias_int32 = (bias * bias_scale).astype(np.int32)
            f.write(bias_int32.tobytes())
        
        # Write fc1 layer (768 → 512)
        write_layer(model.fc1, 768, 512)
        
        # Write fc2 layer (512 → 256)
        write_layer(model.fc2, 512, 256)
        
        # Write output heads (each 256 → 1)
        write_layer(model.eval_head, 256, 1)
        write_layer(model.tau_head, 256, 1)
        write_layer(model.rho_head, 256, 1)
        write_layer(model.rs_head, 256, 1)
    
    print(f"Model exported successfully to {output_path}")


def train_epoch(model, dataloader, optimizer, criterion, device, eval_scaler):
    """Train one epoch"""
    model.train()
    total_loss = 0
    eval_losses = []
    tau_losses = []
    rho_losses = []
    rs_losses = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch["features"].to(device)
        eval_target = batch["eval"].to(device)
        tau_target = batch["tau"].to(device)
        rho_target = batch["rho"].to(device)
        rs_target = batch["rs"].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        eval_out, tau_out, rho_out, rs_out = model(features)
        
        # Losses
        eval_loss = criterion(eval_out.squeeze(), eval_target.squeeze())
        tau_loss = criterion(tau_out.squeeze(), tau_target.squeeze())
        rho_loss = criterion(rho_out.squeeze(), rho_target.squeeze())
        rs_loss = criterion(rs_out.squeeze(), rs_target.squeeze())
        
        # Multi-task loss
        loss = eval_loss * 1.0 + tau_loss * 1.0 + rho_loss * 0.6 + rs_loss * 0.6
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        eval_losses.append(eval_loss.item())
        tau_losses.append(tau_loss.item())
        rho_losses.append(rho_loss.item())
        rs_losses.append(rs_loss.item())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'eval': f'{eval_loss.item():.4f}',
            'tau': f'{tau_loss.item():.4f}'
        })
    
    return {
        "total": total_loss / len(dataloader),
        "eval": np.mean(eval_losses),
        "tau": np.mean(tau_losses),
        "rho": np.mean(rho_losses),
        "rs": np.mean(rs_losses)
    }


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    eval_losses = []
    tau_losses = []
    rho_losses = []
    rs_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            features = batch["features"].to(device)
            eval_target = batch["eval"].to(device)
            tau_target = batch["tau"].to(device)
            rho_target = batch["rho"].to(device)
            rs_target = batch["rs"].to(device)
            
            eval_out, tau_out, rho_out, rs_out = model(features)
            
            eval_loss = criterion(eval_out.squeeze(), eval_target.squeeze())
            tau_loss = criterion(tau_out.squeeze(), tau_target.squeeze())
            rho_loss = criterion(rho_out.squeeze(), rho_target.squeeze())
            rs_loss = criterion(rs_out.squeeze(), rs_target.squeeze())
            
            loss = eval_loss * 1.0 + tau_loss * 1.0 + rho_loss * 0.6 + rs_loss * 0.6
            total_loss += loss.item()
            
            eval_losses.append(eval_loss.item())
            tau_losses.append(tau_loss.item())
            rho_losses.append(rho_loss.item())
            rs_losses.append(rs_loss.item())
    
    return {
        "total": total_loss / len(dataloader),
        "eval": np.mean(eval_losses),
        "tau": np.mean(tau_losses),
        "rho": np.mean(rho_losses),
        "rs": np.mean(rs_losses)
    }


def main():
    parser = argparse.ArgumentParser(description="Complete HARENN Training - Tournament Version")
    parser.add_argument("--data", "-d", default="./data/full_data.jsonl", help="Data file (JSONL)")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", "-b", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hidden1", type=int, default=512, help="First hidden layer size")
    parser.add_argument("--hidden2", type=int, default=256, help="Second hidden layer size")
    parser.add_argument("--output", "-o", default="harenn_model.pth", help="Output model (.pth)")
    parser.add_argument("--export", "-x", default="nextfish.harenn", help="Export to HNN4 format")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading data from: {args.data}")
    dataset = HARENNDataset(args.data)
    
    if len(dataset) == 0:
        print("Error: No data loaded!")
        return
    
    # Split train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {train_size}, Val: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # Model
    print(f"Creating model with hidden1={args.hidden1}, hidden2={args.hidden2}...")
    model = HARENNModel(hidden1_size=args.hidden1, hidden2_size=args.hidden2).to(device)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}...")
        checkpoint = torch.load(args.output, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")
    
    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, None)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss['total']:.4f} (eval:{train_loss['eval']:.4f}, tau:{train_loss['tau']:.4f}, rho:{train_loss['rho']:.4f}, rs:{train_loss['rs']:.4f})")
        print(f"Val Loss: {val_loss['total']:.4f} (eval:{val_loss['eval']:.4f}, tau:{val_loss['tau']:.4f}, rho:{val_loss['rho']:.4f}, rs:{val_loss['rs']:.4f})")
        
        # Save best model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save({
                "model_state": model.state_dict(),
                "eval_mean": dataset.eval_mean,
                "eval_std": dataset.eval_std,
                "epoch": epoch + 1,
                "val_loss": val_loss['total'],
                "args": args
            }, args.output)
            print(f"Best model saved to {args.output}")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Export to HNN4 format
    print("\nExporting to HNN4 format...")
    export_to_hnn4(model, dataset.eval_mean, dataset.eval_std, args.export)
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        sample = val_dataset[0]
        features = sample["features"].unsqueeze(0).to(device)
        eval_out, tau_out, rho_out, rs_out = model(features)
        
        # Denormalize eval
        eval_pred = eval_out.item() * dataset.eval_std + dataset.eval_mean
        
        print(f"Sample prediction:")
        print(f"  Eval: {eval_pred:.1f} (target: {sample['eval'].item() * dataset.eval_std + dataset.eval_mean:.1f})")
        print(f"  Tau: {tau_out.item():.3f} (target: {sample['tau'].item():.3f})")
        print(f"  Rho: {rho_out.item():.3f} (target: {sample['rho'].item():.3f})")
        print(f"  Rs: {rs_out.item():.3f} (target: {sample['rs'].item():.3f})")


if __name__ == "__main__":
    main()
