# Nextfish Data Generation Scripts

This repository contains scripts for generating and processing training data for the HARENN (Heuristic and Reinforcement Enhanced Neural Network) chess engine.

## Scripts

### Training
- **train_harenn_complete.py**: Complete training script for HARENN with 2-layer architecture (768 → 512 → 256 → 4 heads)
  - Supports side-to-move awareness (board flipping for Black)
  - Exports to HNN4 binary format
  - Includes validation and learning rate scheduling

### Data Processing
- **filter_new_data.py**: Filters new GitHub data files that are not in existing dataset
- **merge_new_data.py**: Merges new data files with existing dataset
- **check_fake_labels.py**: Checks for fake/invalid labels in training data
- **filter_fake_labels.py**: Filters out positions with missing eval_score
- **calculate_missing_eval.py**: Calculates missing eval_score labels using chess engine

## Usage

### Training
```bash
python train_harenn_complete.py --data full_data_770k.jsonl --epochs 15 --hidden1 512 --hidden2 256
```

### Data Processing
```bash
# Filter new data
python filter_new_data.py

# Merge data
python merge_new_data.py

# Check for fake labels
python check_fake_labels.py

# Filter fake labels
python filter_fake_labels.py

# Calculate missing eval scores
python calculate_missing_eval.py
```

## Data Format

Training data should be in JSONL format with the following fields:
- `fen`: FEN string
- `eval_score`: Evaluation score in centipawns
- `tau`: Tactical complexity (0-1)
- `rho`: Risk factor (0-1)
- `rs`: Result score (0-1)

## Notes

- The training script includes board flipping for Black to move (Stockfish NNUE style)
- This ensures the model always evaluates from the perspective of the side to move
- The dataset should be cleaned of fake labels before training
