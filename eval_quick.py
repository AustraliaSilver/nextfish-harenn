#!/usr/bin/env python3
"""
Quick eval test - just print eval for a FEN position

Usage:
    python eval_quick.py "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1NR w KQkq - 2 3"
"""

import sys
import torch
import argparse


def fen_to_features(fen: str):
    """Convert FEN to NNUE-style feature vector"""
    parts = fen.split()
    board = parts[0]
    stm = 0 if parts[1] == "w" else 1

    features = torch.zeros(768, dtype=torch.float32)

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


class HARENNNet(torch.nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.input = torch.nn.Linear(769, hidden_size)
        self.hidden1 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.eval_head = torch.nn.Linear(hidden_size // 2, 1)
        self.tau_head = torch.nn.Linear(hidden_size // 2, 1)
        self.result_head = torch.nn.Linear(hidden_size // 2, 1)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)

    def forward(self, x, stm):
        x = torch.cat([x, stm.unsqueeze(1)], dim=1)
        x = self.relu(self.bn1(self.input(x)))
        x = self.relu(self.bn2(self.hidden1(x)))
        x = self.relu(self.hidden2(x))
        eval_out = self.eval_head(x)
        tau_out = torch.sigmoid(self.tau_head(x))
        result_out = torch.sigmoid(self.result_head(x))
        return eval_out, tau_out, result_out


def main():
    parser = argparse.ArgumentParser(description="Quick HARENN Eval")
    parser.add_argument(
        "fen",
        nargs="?",
        default="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1NR w KQkq - 2 3",
    )
    parser.add_argument("--model", "-m", default="harenn_model.pth")

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    checkpoint = torch.load(args.model, weights_only=False)
    model = HARENNNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    eval_mean = checkpoint.get("eval_mean", 0)
    eval_std = checkpoint.get("eval_std", 100)

    # Evaluate
    with torch.no_grad():
        features, stm = fen_to_features(args.fen)
        features = features.unsqueeze(0)
        stm_tensor = torch.tensor([stm], dtype=torch.float32)

        eval_out, tau_out, result_out = model(features, stm_tensor)

        eval_cp = eval_out.item() * eval_std + eval_mean

        print(f"\nPosition: {args.fen[:50]}...")
        print(f"Eval: {eval_cp:.1f} cp")
        print(f"Tau: {tau_out.item():.3f}")
        print(f"Result: {result_out.item():.3f}")


if __name__ == "__main__":
    main()
