#!/usr/bin/env python3
"""
Export HARENN model to C++ readable format
Similar to Stockfish NNUE format

Usage:
    python export_model.py --model harenn_model.pth --output nnue/harenn.nnue
"""

import struct
import torch
import os
import argparse
import json


class HARENNNet(torch.nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.input = torch.nn.Linear(769, hidden_size)
        self.hidden1 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.eval_head = torch.nn.Linear(hidden_size // 2, 1)
        self.tau_head = torch.nn.Linear(hidden_size // 2, 1)
        self.result_head = torch.nn.Linear(hidden_size // 2, 1)
        self.horizon_head = torch.nn.Linear(hidden_size // 2, 1)
        self.resolution_head = torch.nn.Linear(hidden_size // 2, 1)
        self.mcs_h1 = torch.nn.Linear(hidden_size // 2, 64)
        self.mcs_out = torch.nn.Linear(64, 4096)
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
        horizon_out = torch.sigmoid(self.horizon_head(x))
        resolution_out = torch.sigmoid(self.resolution_head(x))
        mcs = self.relu(self.mcs_h1(x))
        mcs_out = torch.sigmoid(self.mcs_out(mcs))
        return eval_out, tau_out, result_out, horizon_out, resolution_out, mcs_out


def export_to_bin(model_path: str, output_path: str):
    """Export model to binary format for C++"""

    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)

    model = HARENNNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Get normalization params
    eval_mean = checkpoint.get("eval_mean", 0)
    eval_std = checkpoint.get("eval_std", 100)

    # Create output directory
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    # Write binary file
    with open(output_path, "wb") as f:
        # Header
        f.write(b"HARN")  # Magic: HARENN
        f.write(struct.pack("i", 2))  # Version

        # Save normalization params
        f.write(struct.pack("f", eval_mean))
        f.write(struct.pack("f", eval_std))

        # Weights: Input layer
        w_input = model.input.weight.detach().numpy()
        b_input = model.input.bias.detach().numpy()
        f.write(struct.pack("i", w_input.shape[0]))  # input_size
        f.write(struct.pack("i", w_input.shape[1]))  # 769
        f.write(w_input.astype("float32").tobytes())
        f.write(b_input.astype("float32").tobytes())

        # Weights: Hidden1
        w_h1 = model.hidden1.weight.detach().numpy()
        b_h1 = model.hidden1.bias.detach().numpy()
        f.write(struct.pack("i", w_h1.shape[0]))
        f.write(struct.pack("i", w_h1.shape[1]))
        f.write(w_h1.astype("float32").tobytes())
        f.write(b_h1.astype("float32").tobytes())

        # Weights: Hidden2
        w_h2 = model.hidden2.weight.detach().numpy()
        b_h2 = model.hidden2.bias.detach().numpy()
        f.write(struct.pack("i", w_h2.shape[0]))
        f.write(struct.pack("i", w_h2.shape[1]))
        f.write(w_h2.astype("float32").tobytes())
        f.write(b_h2.astype("float32").tobytes())

        # Weights: Eval head
        w_eval = model.eval_head.weight.detach().numpy()
        b_eval = model.eval_head.bias.detach().numpy()
        f.write(w_eval.astype("float32").tobytes())
        f.write(b_eval.astype("float32").tobytes())

        # Weights: Tau head
        w_tau = model.tau_head.weight.detach().numpy()
        b_tau = model.tau_head.bias.detach().numpy()
        f.write(w_tau.astype("float32").tobytes())
        f.write(b_tau.astype("float32").tobytes())

        # Weights: Result head
        w_result = model.result_head.weight.detach().numpy()
        b_result = model.result_head.bias.detach().numpy()
        f.write(w_result.astype("float32").tobytes())
        f.write(b_result.astype("float32").tobytes())

        # Weights: Horizon head
        w_hor = model.horizon_head.weight.detach().numpy()
        b_hor = model.horizon_head.bias.detach().numpy()
        f.write(w_hor.astype("float32").tobytes())
        f.write(b_hor.astype("float32").tobytes())

        # Weights: Resolution head
        w_res = model.resolution_head.weight.detach().numpy()
        b_res = model.resolution_head.bias.detach().numpy()
        f.write(w_res.astype("float32").tobytes())
        f.write(b_res.astype("float32").tobytes())

        # Weights: MCS head (256->64->4096)
        w_mcs1 = model.mcs_h1.weight.detach().numpy()
        b_mcs1 = model.mcs_h1.bias.detach().numpy()
        f.write(w_mcs1.astype("float32").tobytes())
        f.write(b_mcs1.astype("float32").tobytes())

        w_mcs2 = model.mcs_out.weight.detach().numpy()
        b_mcs2 = model.mcs_out.bias.detach().numpy()
        f.write(w_mcs2.astype("float32").tobytes())
        f.write(b_mcs2.astype("float32").tobytes())

        # BatchNorm parameters
        # BN1
        f.write(model.bn1.weight.detach().numpy().astype("float32").tobytes())
        f.write(model.bn1.bias.detach().numpy().astype("float32").tobytes())
        f.write(model.bn1.running_mean.detach().numpy().astype("float32").tobytes())
        f.write(model.bn1.running_var.detach().numpy().astype("float32").tobytes())

        # BN2
        f.write(model.bn2.weight.detach().numpy().astype("float32").tobytes())
        f.write(model.bn2.bias.detach().numpy().astype("float32").tobytes())
        f.write(model.bn2.running_mean.detach().numpy().astype("float32").tobytes())
        f.write(model.bn2.running_var.detach().numpy().astype("float32").tobytes())

    # Get file size
    file_size = os.path.getsize(output_path)
    print(f"Exported to: {output_path} ({file_size / 1024:.1f} KB)")

    return file_size


def export_to_onnx(model_path: str, output_path: str):
    """Export model to ONNX format (for C++ inference with onnxruntime)"""

    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)

    model = HARENNNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 768)
    dummy_stm = torch.tensor([0.0])

    # Export
    torch.onnx.export(
        model,
        (dummy_input, dummy_stm),
        output_path,
        input_names=["features", "stm"],
        output_names=["eval", "tau", "result", "horizon", "resolution", "mcs"],
        dynamic_axes={"features": {0: "batch_size"}, "stm": {0: "batch_size"}},
    )

    file_size = os.path.getsize(output_path)
    print(f"ONNX exported to: {output_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HARENN to C++ format")
    parser.add_argument("--model", "-m", default="harenn_model.pth", help="Input model")
    parser.add_argument("--output", "-o", default="harenn.nnue", help="Output file")
    parser.add_argument("--onnx", action="store_true", help="Also export to ONNX")

    args = parser.parse_args()

    # Export to binary
    export_to_bin(args.model, args.output)

    if args.onnx:
        onnx_path = args.output.replace(".nnue", ".onnx")
        export_to_onnx(args.model, onnx_path)

    print("\nDone! To integrate into engine:")
    print(f"1. Binary: {args.output}")
    print("2. Load weights in C++ and use for evaluation")
