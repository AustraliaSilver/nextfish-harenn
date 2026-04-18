#!/usr/bin/env python3
"""
Filter new data files from GitHub that are not in full_data_770k.jsonl
"""
import json
import hashlib
from pathlib import Path

# Load full_data_770k.jsonl FENs to check for duplicates
full_data_path = Path("../downloaded_data/full_data_770k.jsonl")
github_data_dir = Path("../downloaded_data/temp_repo/data")

print("Loading full_data_770k.jsonl FENs...")
full_data_fens = set()
with open(full_data_path, 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            full_data_fens.add(data['fen'])
print(f"Loaded {len(full_data_fens)} unique FENs from full_data_770k.jsonl")

# Check each GitHub file
print("\nChecking GitHub data files...")
new_files = []
for jsonl_file in sorted(github_data_dir.glob("*.jsonl")):
    print(f"\nChecking {jsonl_file.name}...")
    
    new_fens = set()
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                fen = data['fen']
                if fen not in full_data_fens:
                    new_fens.add(fen)
    
    if new_fens:
        print(f"  -> NEW: {len(new_fens)} new FENs")
        new_files.append((jsonl_file.name, len(new_fens)))
    else:
        print(f"  -> Already merged (0 new FENs)")

print("\n" + "="*60)
print("SUMMARY: New files not in full_data_770k.jsonl")
print("="*60)
if new_files:
    total_new = sum(count for _, count in new_files)
    print(f"Total new FENs: {total_new}")
    print("\nFiles:")
    for filename, count in sorted(new_files, key=lambda x: x[1], reverse=True):
        print(f"  {filename}: {count} new FENs")
else:
    print("No new files found - all data already merged")
