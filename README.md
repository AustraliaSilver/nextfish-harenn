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
