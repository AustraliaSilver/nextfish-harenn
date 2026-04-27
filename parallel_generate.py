#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import chess
import chess.engine
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)

@dataclass
class TrainingPosition:
    fen: str
    stm: int
    eval_score: int
    depth: int
    best_move: str
    best_move_label: int
    best_moves_d16: list
    best_moves_d20: list
    best_moves_d24: list
    best_move_labels_d16: list
    best_move_labels_d20: list
    best_move_labels_d24: list
    game_result: int
    material: int
    piece_count: int
    tau: float
    rho: float
    rs: float

class ParallelGenerator:
    def __init__(self, engine_path, output_dir, depth=16):
        self.engine_path = engine_path
        self.output_dir = Path(output_dir)
        self.depth = depth
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def move_to_label(self, board, move):
        if not move or move == chess.Move.null(): return 0
        from_sq = move.from_square
        to_sq = move.to_square
        piece = board.piece_at(from_sq)
        if piece is None: return 0
        fen_sq = to_sq ^ 56
        piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
        p_idx = piece_map[piece.piece_type]
        if piece.color == chess.BLACK: p_idx += 6
        return min(fen_sq * 12 + p_idx, 767)

    def count_material(self, board):
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        total = 0
        for pt, val in vals.items():
            total += len(board.pieces(pt, chess.WHITE)) * val
            total -= len(board.pieces(pt, chess.BLACK)) * val
        return total

    def calculate_rs(self, board, eval_score):
        piece_count = len(board.piece_map())
        if piece_count < 10: return 1.0
        if piece_count < 16: return 0.7
        if not board.is_check() and abs(eval_score) < 30:
            return min(0.8, 0.4 + (30 - abs(eval_score)) / 100.0)
        return 0.1

    def get_top_moves_safe(self, board, engine, depth):
        try:
            res = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
            if not isinstance(res, list): res = [res]
            moves = [r["pv"][0].uci() for r in res if "pv" in r]
            labels = [self.move_to_label(board, r["pv"][0]) for r in res if "pv" in r]
            return moves, labels
        except: return [], []

    def analyze_full(self, board, engine):
        try:
            # Use broader time limits to ensure deep search completes normally, 
            # while still preventing infinite hangs on extremely complex positions.
            res8 = engine.analyse(board, chess.engine.Limit(depth=8, time=1.0))
            res16_multi = engine.analyse(board, chess.engine.Limit(depth=16, time=5.0), multipv=3)
            res26 = engine.analyse(board, chess.engine.Limit(depth=26, time=15.0))
            if not isinstance(res16_multi, list): res16_multi = [res16_multi]
            m8 = res8.get("pv", [None])[0]
            m16 = res16_multi[0].get("pv", [None])[0]
            m26 = res26.get("pv", [None])[0]
            cp = res16_multi[0]["score"].relative.score(mate_score=10000) or 0
            
            # WDL estimation instead of fake label
            try:
                wdl = res16_multi[0]["score"].relative.wdl(model="sf").expectation()
            except:
                wdl = 0.5 + (max(min(cp, 1000), -1000) / 2000.0) # Fallback heuristic

            rho = 1.0 if (m8 != m16 or m16 != m26) else min(abs(cp - (res26["score"].relative.score(mate_score=10000) or 0)) / 200.0 + (0.25 if board.is_check() else 0), 1.0)
            tau = 0.0
            if len(res16_multi) > 1:
                diffs = [abs(cp - (r["score"].relative.score(mate_score=10000) or 0)) for r in res16_multi[1:]]
                tau = min((sum(diffs) / len(diffs)) / 100.0, 1.0)
            m16, l16 = [r["pv"][0].uci() for r in res16_multi], [self.move_to_label(board, r["pv"][0]) for r in res16_multi]
            m20, l20 = self.get_top_moves_safe(board, engine, 20)
            m24, l24 = self.get_top_moves_safe(board, engine, 24)
            return {"cp": cp, "tau": round(tau, 4), "rho": round(min(rho, 1.0), 4), "rs": self.calculate_rs(board, cp), "wdl": round(wdl, 4), "m16": m16, "l16": l16, "m20": m20, "l20": l20, "m24": m24, "l24": l24}
        except: return None

    def worker_loop(self, worker_id, end_time):
        # Seed random with time and worker_id for unique data per matrix job
        random.seed(int(time.time()) + hash(worker_id))
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        total_game_pos = 0
        game_id = 0
        try:
            while time.time() < end_time:
                game_id += 1
                board = chess.Board()
                positions = []
                batch_num = 0
                for move_num in range(1, 100):
                    if time.time() >= end_time: break
                    if board.is_game_over(): break
                    if random.random() < 0.08: move = random.choice(list(board.legal_moves))
                    else: move = engine.play(board, chess.engine.Limit(time=0.04)).move
                    if not move: break
                    board.push(move)
                    if move_num > 6:
                        data = self.analyze_full(board, engine)
                        if data:
                            total_game_pos += 1
                            pos = TrainingPosition(
                                fen=board.fen(), stm=0 if board.turn == chess.WHITE else 1,
                                eval_score=data["cp"], depth=16, best_move=data["m16"][0],
                                best_move_label=self.move_to_label(board, chess.Move.from_uci(data["m16"][0])),
                                best_moves_d16=data["m16"], best_moves_d20=data["m20"], best_moves_d24=data["m24"],
                                best_move_labels_d16=data["l16"], best_move_labels_d20=data["l20"], best_move_labels_d24=data["l24"],
                                game_result=data["wdl"], material=self.count_material(board), piece_count=len(board.piece_map()),
                                tau=data["tau"], rho=data["rho"], rs=data["rs"]
                            )
                            positions.append(asdict(pos))
                            if len(positions) >= 500:
                                filename = f"w{worker_id}_g{game_id}_b{batch_num}_{int(time.time())}.json"
                                with open(self.output_dir / filename, 'w') as f: json.dump({"positions": positions}, f, indent=2)
                                positions, batch_num = [], batch_num + 1
                                print(f"Worker {worker_id} saved large batch {batch_num} (500 pos)", flush=True)
                if positions:
                    filename = f"w{worker_id}_g{game_id}_final_{int(time.time())}.json"
                    with open(self.output_dir / filename, 'w') as f: json.dump({"positions": positions}, f, indent=2)
            return total_game_pos
        finally: engine.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", "-e", default="./harenn_engine")
    parser.add_argument("--output", "-o", default="./data")
    parser.add_argument("--timeout", "-t", type=int, default=18000)
    parser.add_argument("--concurrency", "-j", type=int, default=2)
    args = parser.parse_args()
    end_time = time.time() + args.timeout
    print(f"Starting continuous generation for {args.timeout}s on {args.concurrency} cores...")
    gen = ParallelGenerator(args.engine, args.output)
    with ProcessPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(gen.worker_loop, i, end_time) for i in range(args.concurrency)]
        total = sum(f.result() for f in futures)
    print(f"Total positions generated: {total}")
