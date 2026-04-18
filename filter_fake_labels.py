#!/usr/bin/env python3
"""
Filter out fake labels (missing eval_score) from the data
"""
import json
from pathlib import Path

data_path = Path("../downloaded_data/full_data_770k.jsonl")
output_path = Path("../downloaded_data/full_data_clean.jsonl")

print("Loading data...")
lines = []
with open(data_path, 'r') as f:
    for line in f:
        if line.strip():
            lines.append(json.loads(line))

print(f"Loaded {len(lines)} lines")

# Filter out fake labels (missing eval_score)
print("\nFiltering out fake labels...")
clean_lines = []
for data in lines:
    eval_score = data.get('eval_score')
    if eval_score is not None:
        clean_lines.append(data)

print(f"Filtered out {len(lines) - len(clean_lines)} lines")
print(f"Remaining: {len(clean_lines)} lines ({len(clean_lines)/len(lines)*100:.2f}%)")

# Write clean data
print(f"\nWriting clean data to {output_path}...")
with open(output_path, 'w') as f:
    for data in clean_lines:
        f.write(json.dumps(data) + '\n')

print("Done!")
