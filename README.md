<<<<<<< Updated upstream
# nextfish-harenn

Automated data generation for training HARENN (HARE-Integrated Neural Network).

## What this repo now does

- Generates opening seeds from PGN using `python-chess` (`preprocess_pgn.py`).
- Runs self-play label generation with Stockfish backend (`harenn_gen.cpp`).
- Orchestrates multi-worker generation, shard validation, merge, compression and manifest export (`scripts/generate_dataset.py`).
- Runs automatically on GitHub Actions and uploads ready-to-train artifacts.

## Local usage

1. Build generator:

```bash
g++ -O3 -std=c++17 harenn_gen.cpp -o harenn_gen -lpthread
```

2. Build opening seeds:

```bash
python preprocess_pgn.py --input books/UHO_2022_8mvs_+110_+119.pgn --output book_moves.txt --lines 8000 --plies 8 --strict-source
```

3. Run orchestrator:

```bash
python scripts/generate_dataset.py \
  --repo-root . \
  --generator ./harenn_gen \
  --stockfish ./stockfish \
  --games 120 \
  --workers 2 \
  --out-dir data/generated \
  --prefix harenn \
  --strict-quality
```

Outputs:
- `*.binpack` merged dataset
- `*.train.binpack`, `*.val.binpack`, `*.test.binpack` deterministic splits
- `*.hard.binpack` hard subset for tactical/horizon-focused training
- `*.binpack.zst` (or `.gz`) compressed dataset
- `*.manifest.json` with quality stats and checksums

## Notes

- Generator now uses fixed Stockfish options for reproducibility: `Threads=1`, `Hash=256`.
- Repeated positions in a game are deduplicated before writing labels.
- Datagen enforces quality gates on merged and splits (entry count, label ranges, MCS density, phase diversity).
- In strict mode, opening PGN must exist and parse successfully (no random fallback).
- Workflow uploads artifacts instead of committing large binary files into git history.
=======
# HARENN Training Data Generation

This directory contains the training data generation pipeline for HARENN (HARE-Integrated Neural Network).

## Overview

HARENN is a multi-output neural network that outputs 5 different values:
1. **Eval** (standard NNUE evaluation)
2. **Tactical Complexity (τ)** - How tactically complex is the position?
3. **Move Criticality Scores (MCS)** - 64x64 map of how critical each move is
4. **Horizon Risk (ρ)** - Risk of horizon effect
5. **Resolution Score (rs)** - How much the position needs deeper search

## Files

- `generate_training_data.py` - Main training data generation script
- `collect_data_cutechess.py` - Alternative data collection using cutechess-cli
- `requirements.txt` - Python dependencies

## Usage

### Basic Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data
python generate_training_data.py --engine ./nextfish.exe --games 1000 --output ./data
```

### Options

- `--engine, -e`: Path to chess engine (default: ./nextfish.exe)
- `--output, -o`: Output directory (default: ./data)
- `--games, -g`: Number of self-play games (default: 100)
- `--workers, -w`: Number of parallel workers (default: 4)

## GitHub Workflows

### harenn-training.yml

This workflow runs automatically when:
- Code is pushed to main/master or harenn/** branches
- Changes to Python files or documentation

It generates training data using multiple parallel jobs and merges the results.

### harenn-test.yml

This workflow runs tests and validation:
- Script smoke tests
- Data quality validation
- Optional Elo testing (manual trigger)

## Data Format

The generated binary files contain:
- Position features (HalfKAv2 format)
- Side to move
- Game result
- All 5 labels (eval, tactical, MCS, horizon, resolution)

## Quality Guidelines

For high-quality training data:
1. Use diverse opening book (UHO recommended)
2. Generate at least 1000 games per dataset
3. Use moderate time controls (10+0.1 or longer)
4. Include both engine self-play and engine vs Stockfish games
5. Sample positions throughout the game, not just endgame

## References

See `../New folder (4)/HARENN.md` for the full HARENN architecture specification.
>>>>>>> Stashed changes
