#!/usr/bin/env python3
"""
Calculate missing eval_score labels using chess engine
"""
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# Paths
data_with_fake_path = Path("../downloaded_data/full_data_with_fake.jsonl")
output_path = Path("../downloaded_data/full_data_complete.jsonl")
engine_path = Path("../src/stockfish.exe")

print("Loading data with fake labels...")
lines = []
with open(data_with_fake_path, 'r') as f:
    for line in f:
        if line.strip():
            lines.append(json.loads(line))

print(f"Loaded {len(lines)} lines")

# Find positions with missing eval_score
print("\nFinding positions with missing eval_score...")
missing_eval_positions = []
for i, data in enumerate(lines):
    if data.get('eval_score') is None:
        missing_eval_positions.append(i)

print(f"Found {len(missing_eval_positions)} positions with missing eval_score")

if len(missing_eval_positions) == 0:
    print("No missing eval_score found. Nothing to do.")
    exit(0)

# Start engine
print(f"\nStarting engine: {engine_path}")
engine = subprocess.Popen(
    [str(engine_path)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Send UCI commands
engine.stdin.write("uci\n")
engine.stdin.flush()

# Wait for uciok
while True:
    line = engine.stdout.readline()
    if "uciok" in line:
        break

def evaluate_position(fen):
    """Evaluate a position using the engine (faster with eval command)"""
    engine.stdin.write(f"position fen {fen}\n")
    engine.stdin.write("eval\n")
    engine.stdin.flush()
    
    while True:
        line = engine.stdout.readline()
        if line.startswith("score"):
            # Parse score from eval output
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "mate":
                    # Convert mate to centipawns (very large value)
                    mate_in = int(parts[i+1])
                    return 100000 * mate_in
                elif part == "cp":
                    return int(parts[i+1])
            # If no specific score type, return 0
            return 0

# Calculate missing eval scores
print(f"\nCalculating eval_score for {len(missing_eval_positions)} positions...")
print("This may take a while...")

for idx in tqdm(missing_eval_positions):
    data = lines[idx]
    fen = data['fen']
    
    try:
        eval_score = evaluate_position(fen)
        data['eval_score'] = eval_score
    except Exception as e:
        print(f"\nError evaluating position {idx}: {e}")
        data['eval_score'] = 0  # Default to 0 on error

# Write complete data
print(f"\nWriting complete data to {output_path}...")
with open(output_path, 'w') as f:
    for data in lines:
        f.write(json.dumps(data) + '\n')

# Stop engine
engine.stdin.write("quit\n")
engine.stdin.flush()

print("Done!")
print(f"Total positions: {len(lines)}")
print(f"Positions with eval_score: {sum(1 for d in lines if d.get('eval_score') is not None)}")
