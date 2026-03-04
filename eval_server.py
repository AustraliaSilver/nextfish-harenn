#!/usr/bin/env python3
"""
HARENN Evaluation Server

Provides evaluation via UCI commands.
Run this before your engine, then configure engine to use this as eval provider.

Usage:
    python eval_server.py --model harenn_model.pth

Then in your engine or testing framework, use UCI "eval" command.
"""

import argparse
import torch
import chess
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


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
            if char in piece_map:
                features[square * 12 + piece_map[char]] = 1.0
            square += 1

    return features, stm


class HARENNNet(torch.nn.Module):
    """HARENN Network"""

    def __init__(self, hidden_size=512):
        super().__init__()
        self.input = torch.nn.Linear(769, hidden_size)
        self.hidden1 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.eval_head = torch.nn.Linear(hidden_size // 2, 1)
        self.tau_head = torch.nn.Linear(hidden_size // 2, 1)
        self.result_head = torch.nn.Linear(hidden_size // 2, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, stm):
        x = torch.cat([x, stm.unsqueeze(1)], dim=1)
        x = self.relu(self.input(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        eval_out = self.eval_head(x)
        tau_out = torch.sigmoid(self.tau_head(x))
        result_out = torch.sigmoid(self.result_head(x))
        return eval_out, tau_out, result_out


class EvalHandler(BaseHTTPRequestHandler):
    """HTTP handler for eval requests"""

    def do_POST(self):
        if self.path == "/eval":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode())
                fen = data.get("fen", "")

                if fen:
                    result = evaluate_position(fen)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                    return
            except:
                pass

            self.send_response(400)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging


model = None
eval_mean = 0
eval_std = 100


def load_model(model_path: str):
    """Load trained model"""
    global model, eval_mean, eval_std

    checkpoint = torch.load(model_path, weights_only=False)
    model = HARENNNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    eval_mean = checkpoint.get("eval_mean", 0)
    eval_std = checkpoint.get("eval_std", 100)

    print(f"Model loaded: {model_path}")
    print(f"Eval normalization: mean={eval_mean:.2f}, std={eval_std:.2f}")


def evaluate_position(fen: str) -> dict:
    """Evaluate a position using HARENN"""
    global model, eval_mean, eval_std

    if model is None:
        return {"error": "Model not loaded"}

    with torch.no_grad():
        features, stm = fen_to_features(fen)
        features = features.unsqueeze(0)
        stm_tensor = torch.tensor([stm], dtype=torch.float32)

        eval_out, tau_out, result_out = model(features, stm_tensor)

        # Denormalize eval
        eval_cp = eval_out.item() * eval_std + eval_mean

        return {
            "eval": round(eval_cp, 1),
            "tau": round(tau_out.item(), 3),
            "result": round(result_out.item(), 3),
            "fen": fen[:60],
        }


def start_server(port: int = 5555):
    """Start eval server"""
    server = HTTPServer(("localhost", port), EvalHandler)
    print(f"HARENN Eval Server running on http://localhost:{port}")
    print(
        f"POST /eval with {{'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1NR w KQkq - 2 3'}}"
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HARENN Eval Server")
    parser.add_argument("--model", "-m", default="harenn_model.pth", help="Model file")
    parser.add_argument("--port", "-p", type=int, default=5555, help="Server port")

    args = parser.parse_args()

    load_model(args.model)
    start_server(args.port)
