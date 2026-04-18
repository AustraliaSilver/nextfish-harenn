#!/usr/bin/env python3
"""
Merge new GitHub data files with full_data_770k.jsonl
"""
import json
from pathlib import Path

# Paths
full_data_path = Path("../downloaded_data/full_data_770k.jsonl")
github_data_dir = Path("../downloaded_data/temp_repo/data")
output_path = Path("../downloaded_data/full_data_merged.jsonl")

# New files to merge
new_files = [
    "harenn_standard_24543786092.jsonl",
    "harenn_standard_24574177424.jsonl",
    "harenn_standard_24552152397.jsonl",
    "harenn_standard_24571978999.jsonl",
    "harenn_standard_24561282619.jsonl",
    "harenn_standard_24585659378.jsonl"
]

# Load existing FENs to avoid duplicates
print("Loading full_data_770k.jsonl FENs...")
existing_fens = set()
existing_data = []
with open(full_data_path, 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            existing_fens.add(data['fen'])
            existing_data.append(data)
print(f"Loaded {len(existing_data)} lines from full_data_770k.jsonl")

# Merge new data
print("\nMerging new GitHub data files...")
new_data = []
for filename in new_files:
    filepath = github_data_dir / filename
    print(f"  Processing {filename}...")
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                fen = data['fen']
                if fen not in existing_fens:
                    new_data.append(data)
                    existing_fens.add(fen)

print(f"Added {len(new_data)} new lines")

# Write merged data
print(f"\nWriting merged data to {output_path}...")
with open(output_path, 'w') as f:
    for data in existing_data:
        f.write(json.dumps(data) + '\n')
    for data in new_data:
        f.write(json.dumps(data) + '\n')

total_lines = len(existing_data) + len(new_data)
print(f"Done! Total lines: {total_lines}")
