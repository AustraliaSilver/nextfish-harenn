import json
import os
from pathlib import Path

data_dir = Path("./data_batches")
output_file = Path("./merged_data.jsonl")

all_positions = []

for batch_dir in data_dir.iterdir():
    if batch_dir.is_dir():
        json_files = list(batch_dir.glob("harenn_positions_*.json"))
        if json_files:
            with open(json_files[0], "r") as f:
                data = json.load(f)
                if "positions" in data:
                    all_positions.extend(data["positions"])

# Shuffle and save
import random

random.shuffle(all_positions)

with open(output_file, "w") as f:
    for pos in all_positions:
        f.write(json.dumps(pos) + "\n")

print(f"Merged {len(all_positions)} positions into {output_file}")
