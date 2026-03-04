#!/usr/bin/env python3
"""
HARENN Training Data Generation Pipeline (Updated for Standard Data)
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
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import threading
import queue

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import chess
    import chess.pgn
    import chess.engine
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-chess", "numpy"])
    import chess
    import chess.pgn
    import chess.engine

def open_engine(engine_path: str):
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    return eng

@dataclass
class EvalLabel:
    score: int
    depth: int
    best_move: str
    pv: List[str]

@dataclass
class TacticalLabel:
    tau: float
    score_volatility: float
    research_rate: float
    pv_tactical_density: float
    convergence_factor: float

@dataclass
class MCSLabel:
    mcs_map: np.ndarray

@dataclass
class HorizonLabel:
    rho: float
    depth_diff: int
    tactical_estimate: float

@dataclass
class ResolutionLabel:
    rs: float
    qsearch_score_diff: float

@dataclass
class HARENNTrainingSample:
    white_features: bytes
    black_features: bytes
    stm: int
    game_result: int
    eval_label: EvalLabel
    tactical_label: TacticalLabel
    mcs_label: MCSLabel
    horizon_label: HorizonLabel
    resolution_label: ResolutionLabel
    fen: str
    move_number: int
    plies_from_start: int

class HARENNDataGenerator:
    def __init__(self, engine_path: str, output_dir: str, num_workers: int = 4):
        self.engine_path = engine_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"positions_generated": 0, "games_played": 0, "errors": 0}

    def generate_labels_from_trace(self, board: chess.Board) -> Optional[Dict]:
        try:
            with open_engine(self.engine_path) as engine:
                # Using lower level access to get raw output
                # This depends on how the engine responds to 'eval'
                command = f"position fen {board.fen()}\neval\n"
                
                # Simplified: we'll use analyse with a very short limit
                # and assume our C++ changes output info string with labels
                limit = chess.engine.Limit(time=0.01)
                # But since Eval::trace is stdout, we might need to use popen directly
                
            import subprocess
            proc = subprocess.Popen([self.engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            proc.stdin.write(f"position fen {board.fen()}\neval\nquit\n")
            proc.stdin.flush()
            
            output = proc.stdout.read()
            proc.wait()
            
            labels = {}
            for line in output.split('\n'):
                if "HARENN_TAU:" in line: labels['tau'] = float(line.split(':')[1].strip())
                elif "HARENN_RHO:" in line: labels['rho'] = float(line.split(':')[1].strip())
                elif "HARENN_RS:" in line: labels['rs'] = float(line.split(':')[1].strip())
                elif "DEE_SCORE:" in line: labels['dee_score'] = float(line.split(':')[1].strip())
                elif "DEE_THREAT:" in line: labels['dee_threat'] = float(line.split(':')[1].strip())
            
            return labels if 'tau' in labels else None
        except Exception as e:
            return None

    def generate_sample(self, board: chess.Board, game_result: int) -> Optional[HARENNTrainingSample]:
        if board.ply() < 12 or len(board.piece_map()) < 10:
            return None
            
        trace_labels = self.generate_labels_from_trace(board)
        if not trace_labels:
            return None

        self.stats["positions_generated"] += 1
        
        # Construct placeholders for missing bits
        eval_l = EvalLabel(score=int(trace_labels.get('dee_score', 0)), depth=0, best_move="0000", pv=[])
        tac_l = TacticalLabel(tau=trace_labels['tau'], score_volatility=0, research_rate=0, pv_tactical_density=0, convergence_factor=0)
        mcs_l = MCSLabel(mcs_map=np.zeros((64,64)))
        hor_l = HorizonLabel(rho=trace_labels['rho'], depth_diff=0, tactical_estimate=0)
        res_l = ResolutionLabel(rs=trace_labels['rs'], qsearch_score_diff=0)

        return HARENNTrainingSample(
            white_features=b"", black_features=b"", stm=int(board.turn), game_result=game_result,
            eval_label=eval_l, tactical_label=tac_l, mcs_label=mcs_l, horizon_label=hor_l, resolution_label=res_l,
            fen=board.fen(), move_number=board.fullmove_number, plies_from_start=board.ply()
        )

    def generate_game_data(self, num_games: int, output_file: str, epd_file: Optional[str] = None):
        print(f"Generating {num_games} games...")
        
        start_positions = []
        if epd_file and os.path.exists(epd_file):
            print(f"Loading openings from {epd_file}...")
            with open(epd_file, "r") as f:
                start_positions = [line.strip() for line in f if line.strip()]
        
        jsonl_path = self.output_dir / output_file
        with open(jsonl_path, "a") as f:
            with open_engine(self.engine_path) as engine:
                for g_idx in range(num_games):
                    board = chess.Board()
                    if start_positions:
                        fen = random.choice(start_positions)
                        try:
                            board.set_epd(fen)
                        except Exception:
                            board.set_fen(fen) # Fallback to FEN if EPD parse fails

                    while not board.is_game_over() and board.ply() < 150:
                        res = engine.play(board, chess.engine.Limit(time=0.05))
                        board.push(res.move)
                        if random.random() < 0.2:
                            sample = self.generate_sample(board, 1) 
                            if sample:
                                item = {"fen": sample.fen, "tau": sample.tactical_label.tau, "rho": sample.horizon_label.rho, "rs": sample.resolution_label.rs}
                                f.write(json.dumps(item) + "\n")
                    self.stats["games_played"] += 1
                    if (g_idx + 1) % 5 == 0:
                        print(f" Game {g_idx+1}/{num_games} done. Positions: {self.stats['positions_generated']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="./stockfish.exe")
    parser.add_argument("--output", default="./data")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--epd", default=None, help="EPD file for start positions")
    args = parser.parse_args()

    gen = HARENNDataGenerator(args.engine, args.output)
    gen.generate_game_data(args.games, "harenn_standard.jsonl", epd_file=args.epd)

if __name__ == "__main__":
    main()
