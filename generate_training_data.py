#!/usr/bin/env python3
"""
HARENN Training Data Generation Pipeline (Updated for Standard Data & Validation)
"""

import argparse
import os
import sys
import json
import time
import random
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

try:
    import chess
    import chess.engine
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-chess", "numpy"])
    import chess
    import chess.engine

def open_engine(engine_path: str):
    return chess.engine.SimpleEngine.popen_uci(engine_path)

@dataclass
class TacticalLabel:
    tau: float
    score_volatility: float
    research_rate: float
    pv_tactical_density: float
    convergence_factor: float

@dataclass
class HARENNTrainingSample:
    fen: str
    stm: int
    game_result: int
    tactical_label: TacticalLabel
    horizon_label_rho: float
    resolution_label_rs: float

class HARENNDataGenerator:
    def __init__(self, engine_path: str, output_dir: str):
        self.engine_path = engine_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"positions_generated": 0, "games_played": 0, "errors": 0}

    def generate_labels_from_trace(self, board: chess.Board) -> Optional[Dict]:
        try:
            import subprocess
            # Use smaller hash to save memory on CI runners
            proc = subprocess.Popen([self.engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            proc.stdin.write(f"setoption name Hash value 16\nposition fen {board.fen()}\neval\nquit\n")
            proc.stdin.flush()
            
            output = proc.stdout.read()
            proc.wait()
            
            labels = {}
            for line in output.split('\n'):
                try:
                    if "HARENN_TAU:" in line: labels['tau'] = float(line.split(':')[1].strip())
                    elif "HARENN_RHO:" in line: labels['rho'] = float(line.split(':')[1].strip())
                    elif "HARENN_RS:" in line: labels['rs'] = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    continue
            
            # Validation: Ensure standard data integrity
            required = ['tau', 'rho', 'rs']
            if all(k in labels for k in required):
                # Basic range checks for standard data
                if all(0 <= labels[k] <= 1.01 for k in required): # Allow small floating point margin
                    return labels
                else:
                    print(f"Validation Error: Labels out of range [0,1] for FEN {board.fen()} : {labels}")
            
            return None
        except Exception as e:
            print(f"Engine trace fatal error: {e}")
            return None

    def generate_sample(self, board: chess.Board, game_result: int) -> Optional[HARENNTrainingSample]:
        if board.ply() < 8 or len(board.piece_map()) < 6:
            return None
            
        trace_labels = self.generate_labels_from_trace(board)
        if not trace_labels:
            return None

        self.stats["positions_generated"] += 1
        
        tac_l = TacticalLabel(tau=trace_labels['tau'], score_volatility=0, research_rate=0, pv_tactical_density=0, convergence_factor=0)

        return HARENNTrainingSample(
            fen=board.fen(), stm=int(board.turn), game_result=game_result,
            tactical_label=tac_l, horizon_label_rho=trace_labels['rho'], resolution_label_rs=trace_labels['rs']
        )

    def generate_game_data(self, num_games: int, output_file: str, epd_file: Optional[str] = None):
        print(f"Generating {num_games} games using Validated Trace Pipeline...")
        
        start_positions = []
        if epd_file and os.path.exists(epd_file):
            with open(epd_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # Strict 8-column check per rank
                    try:
                        fen_part = line.split(' ')[0]
                        ranks = fen_part.split('/')
                        if len(ranks) != 8: continue
                        
                        is_valid_structure = True
                        for r in ranks:
                            count = 0
                            for char in r:
                                if char.isdigit(): count += int(char)
                                else: count += 1
                            if count != 8:
                                is_valid_structure = False
                                break
                        
                        if is_valid_structure:
                            start_positions.append(line)
                    except Exception:
                        continue
            print(f"Loaded {len(start_positions)} strictly valid openings from {epd_file}")
        
        jsonl_path = self.output_dir / output_file
        with open(jsonl_path, "a") as f:
            for g_idx in range(num_games):
                try:
                    with open_engine(self.engine_path) as engine:
                        engine.configure({"Hash": 16})
                        board = chess.Board()
                        if start_positions:
                            fen = random.choice(start_positions)
                            try: board.set_epd(fen)
                            except: board.set_fen(fen)

                        while not board.is_game_over() and board.ply() < 150:
                            res = engine.play(board, chess.engine.Limit(time=0.05))
                            board.push(res.move)
                            if random.random() < 0.25: # Sample ~25% of positions
                                sample = self.generate_sample(board, 1) 
                                if sample:
                                    item = {
                                        "fen": board.fen(), # Always use board.fen() directly
                                        "tau": sample.tactical_label.tau, 
                                        "rho": sample.horizon_label_rho, 
                                        "rs": sample.resolution_label_rs
                                    }
                                    f.write(json.dumps(item) + "\n")
                                    f.flush()
                    self.stats["games_played"] += 1
                    if (g_idx + 1) % 5 == 0:
                        print(f" Progress: {g_idx+1}/{num_games} games. Valid Positions: {self.stats['positions_generated']}")
                except Exception as e:
                    print(f"Game {g_idx} skip due to error: {e}")
                    continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="./stockfish.exe")
    parser.add_argument("--output", default="./data")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--epd", default=None)
    args = parser.parse_args()

    gen = HARENNDataGenerator(args.engine, args.output)
    gen.generate_game_data(args.games, "harenn_standard.jsonl", epd_file=args.epd)

if __name__ == "__main__":
    main()
