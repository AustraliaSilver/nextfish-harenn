#!/usr/bin/env python3
"""
HARENN Training Data Generation Pipeline (V2 - High Quality Tactical Labels)
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path
from typing import Optional
import chess
import chess.engine

class PositionAnalyzer:
    @staticmethod
    def calculate_labels(board: chess.Board):
        us = board.turn
        them = not us
        
        # 1. Tau (Tactical Complexity)
        legal_moves = list(board.legal_moves)
        mobility = len(legal_moves) / 60.0
        
        captures = [m for m in legal_moves if board.is_capture(m)]
        capture_score = len(captures) / 15.0
        
        our_king_sq = board.king(us)
        enemy_attackers = board.attackers(them, our_king_sq)
        king_pressure = len(enemy_attackers) / 4.0
        
        tension = 0
        for sq, piece in board.piece_map().items():
            if piece.color == us:
                attackers = board.attackers(them, sq)
                if attackers:
                    defenders = board.attackers(us, sq)
                    if len(attackers) > len(defenders):
                        tension += piece.piece_type 
                    elif len(attackers) > 0:
                        tension += 0.5
        tension_score = min(1.0, tension / 10.0)
        
        tau = (mobility * 0.2) + (capture_score * 0.3) + (king_pressure * 0.2) + (tension_score * 0.3)
        tau = min(1.0, max(0.0, tau))

        # 2. Rho (Horizon Risk)
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        rho = min(1.0, (queens * 0.3 + rooks * 0.15 + 0.2))

        # 3. Rs (Resolution/Phase)
        piece_count = len(board.piece_map())
        rs = 1.0 - (piece_count / 32.0)

        return round(tau, 4), round(rho, 4), round(rs, 4)

def open_engine(engine_path: str):
    return chess.engine.SimpleEngine.popen_uci(engine_path)

class HARENNDataGenerator:
    def __init__(self, engine_path: str, output_dir: str):
        self.engine_path = engine_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"positions_generated": 0, "games_played": 0}

    def generate_game_data(self, num_games: int, output_file: str, epd_file: Optional[str] = None):
        start_positions = []
        if epd_file and os.path.exists(epd_file):
            with open(epd_file, "r") as f:
                for line in f:
                    if line.strip(): start_positions.append(line.strip())

        jsonl_path = self.output_dir / output_file
        
        with open_engine(self.engine_path) as engine:
            for g_idx in range(num_games):
                try:
                    board = chess.Board()
                    if start_positions:
                        fen = random.choice(start_positions)
                        try: board.set_epd(fen)
                        except: board.set_fen(fen)

                    while not board.is_game_over() and board.ply() < 160:
                        res = engine.play(board, chess.engine.Limit(time=0.02))
                        board.push(res.move)
                        
                        if board.ply() > 10 and random.random() < 0.20:
                            tau, rho, rs = PositionAnalyzer.calculate_labels(board)
                            info = engine.analyse(board, chess.engine.Limit(depth=10))
                            score = info["score"].relative.score(mate_score=10000)
                            
                            item = {
                                "fen": board.fen(),
                                "tau": tau,
                                "rho": rho,
                                "rs": rs,
                                "eval_score": score
                            }
                            with open(jsonl_path, "a") as f:
                                f.write(json.dumps(item) + "\n")
                            self.stats["positions_generated"] += 1
                            
                    self.stats["games_played"] += 1
                except Exception as e:
                    continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="./harenn_engine")
    parser.add_argument("--output", default="./data")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--epd", default=None)
    args = parser.parse_args()
    
    gen = HARENNDataGenerator(args.engine, args.output)
    gen.generate_game_data(args.games, "harenn_standard.jsonl", epd_file=args.epd)

if __name__ == "__main__":
    main()
