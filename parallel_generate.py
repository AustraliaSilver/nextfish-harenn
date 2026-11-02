#!/usr/bin/env python3

"""

HARENN Turbo Parallel Data Generator - V8 (Raw Data Collection)

D:/nextfish/data - Loại bỏ mọi bộ lọc, phân tích 100% nước đi

"""



import argparse

import os

import sys

import json

import time
import uuid

import random

import logging

from pathlib import Path

from dataclasses import dataclass, asdict

import chess

import chess.engine

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

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

    def __init__(self, engine_path, output_dir, nodes=16384, hash_mb=2048, time_budget=None, worker_id=None, batch_size=100):
        self.batch_size = batch_size
        self.worker_id = worker_id
        self.time_budget = time_budget
        self.start_time = time.time()
        self.nodes = nodes
        self.hash_mb = hash_mb
        self.nodes_shallow = max(512, nodes // 16)
        self.nodes_mid = max(1024, nodes // 4)

        self.engine_path = engine_path

        self.output_dir = Path(output_dir)

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
            res_shallow = engine.analyse(board, chess.engine.Limit(nodes=self.nodes_shallow))
            res_mid_multi = engine.analyse(board, chess.engine.Limit(nodes=self.nodes_mid), multipv=2)
            res_deep = engine.analyse(board, chess.engine.Limit(nodes=self.nodes))

            if not isinstance(res_mid_multi, list):
                res_mid_multi = [res_mid_multi]

            m_shallow = res_shallow.get("pv", [None])[0]
            m_mid = res_mid_multi[0].get("pv", [None])[0]
            m_deep = res_deep.get("pv", [None])[0]

            cp_deep = res_deep["score"].relative.score(mate_score=10000) or 0
            cp_mid = res_mid_multi[0]["score"].relative.score(mate_score=10000) or 0
            cp_shallow = res_shallow["score"].relative.score(mate_score=10000) or 0
            cp = cp_deep

            # Eval swing between node budgets (absolute, in centipawns)
            swing_shallow_mid = abs(cp_shallow - cp_mid)
            swing_mid_deep = abs(cp_mid - cp_deep)
            max_swing = max(swing_shallow_mid, swing_mid_deep)

            # RHO (horizon risk): how much the eval moves between node budgets.
            # Normalized so that 200 cp of swing maps to 1.0.
            rho = min(max_swing / 200.0, 1.0)

            # TAU (tactical complexity): how often the best move changes between node budgets.
            transitions = 0
            changes = 0
            if m_shallow is not None and m_mid is not None:
                transitions += 1
                if m_shallow != m_mid:
                    changes += 1
            if m_mid is not None and m_deep is not None:
                transitions += 1
                if m_mid != m_deep:
                    changes += 1
            tau = (changes / transitions) if transitions > 0 else 0.0

            # Bonus for near-zero eval (positions where the outcome is unclear)
            if abs(cp) < 30:
                rho = min(rho + 0.15, 1.0)
                tau = min(tau + 0.15, 1.0)
            elif abs(cp) < 80:
                rho = min(rho + 0.05, 1.0)

            # Bonus for check (tactical positions)
            if board.is_check():
                rho = min(rho + 0.10, 1.0)
                tau = min(tau + 0.10, 1.0)

            # RS (Confidence Score / Resolution): Measures how confident the
            # engine is in its evaluation and move choice compared to potential error/deviation.
            # 1. Budget Stability: Eval deviation across shallow, mid, and deep node budgets
            max_eval_dev = max(abs(cp_shallow - cp_mid), abs(cp_mid - cp_deep), abs(cp_shallow - cp_deep))
            s_stability = max(0.0, 1.0 - (max_eval_dev / 150.0))

            # 2. Move Consistency: Agreement of best move across node budgets
            if m_shallow is not None and m_mid is not None and m_deep is not None:
                if m_shallow == m_mid == m_deep:
                    s_agreement = 1.0
                elif m_mid == m_deep:
                    s_agreement = 0.75
                elif m_shallow == m_deep:
                    s_agreement = 0.50
                else:
                    s_agreement = 0.25
            else:
                s_agreement = 0.50

            # 3. Decision Margin: Dominance of best move over second best move in MultiPV
            if len(res_mid_multi) >= 2 and 'score' in res_mid_multi[1] and res_mid_multi[1].get('pv'):
                cp_second = res_mid_multi[1]['score'].relative.score(mate_score=10000) or 0
                margin = max(0, cp_mid - cp_second)
                s_margin = min(1.0, margin / 80.0)
            else:
                s_margin = 0.50

            # Composite confidence score
            rs = 0.45 * s_stability + 0.35 * s_agreement + 0.20 * s_margin
            rs = max(0.05, min(0.99, rs))

            m_mid_uci = [r["pv"][0].uci() for r in res_mid_multi if r.get("pv")]
            l_mid_lbl = [self.move_to_label(board, r["pv"][0]) for r in res_mid_multi if r.get("pv")]
            m_deep_uci = [m_deep.uci()] if m_deep else []
            l_deep_lbl = [self.move_to_label(board, m_deep)] if m_deep else []
            achieved_depth = int(res_deep.get("depth", 24))

            return {
                "cp": cp,
                "depth": achieved_depth,
                "tau": round(tau, 4),
                "rho": round(min(rho, 1.0), 4),
                "rs": round(rs, 4),
                "m16": m_mid_uci,
                "l16": l_mid_lbl,
                "m20": m_deep_uci,
                "l20": l_deep_lbl,
                "m24": m_deep_uci,
                "l24": l_deep_lbl,
            }

        except Exception:
            return None

    def worker_task(self, game_id):

        p_name = multiprocessing.current_process().name

        worker_id = p_name.split('-')[-1] if '-' in p_name else "0"

        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        try:
            engine.configure({"Threads": 1, "Hash": self.hash_mb, "Move Overhead": 0})
        except Exception:
            pass

        positions, batch_num, board = [], 0, chess.Board()

        total_game_pos = 0

        try:

            if self.time_budget and (time.time() - self.start_time) > self.time_budget:
                print(f"  [W{worker_id} | G{game_id}] Time budget exceeded, stopping", flush=True)
                return total_game_pos
            for move_num in range(1, 100):
                if self.time_budget and (time.time() - self.start_time) > self.time_budget:
                    print(f"  [W{worker_id} | G{game_id}] Time budget reached mid-game, finishing game", flush=True)
                    break

                if board.is_game_over(): break

                if random.random() < 0.08:

                    move = random.choice(list(board.legal_moves))

                else:

                    move = engine.play(board, chess.engine.Limit(time=0.04)).move

                board.push(move)

                

                # BỎ LỌC, LẤY TẤT CẢ SAU NƯỚC 6

                if move_num > 6:

                    data = self.analyze_full(board, engine)

                    if data:

                        total_game_pos += 1

                        print(f"  [W{worker_id} | G{game_id}] Pos {total_game_pos}: Eval {data['cp']}", flush=True)

                        chosen_best = data["m20"][0] if data["m20"] else (data["m16"][0] if data["m16"] else "0000")
                        chosen_best_obj = chess.Move.from_uci(chosen_best) if chosen_best != "0000" else None
                        pos = TrainingPosition(
                            fen=board.fen(), stm=0 if board.turn == chess.WHITE else 1,
                            eval_score=data["cp"], depth=data.get("depth", 24),
                            best_move=chosen_best,
                            best_move_label=self.move_to_label(board, chosen_best_obj) if chosen_best_obj else 0,
                            best_moves_d16=data["m16"], best_moves_d20=data["m20"], best_moves_d24=data["m24"],
                            best_move_labels_d16=data["l16"], best_move_labels_d20=data["l20"], best_move_labels_d24=data["l24"],
                            game_result=1, material=self.count_material(board), piece_count=len(board.piece_map()),
                            tau=data["tau"], rho=data["rho"], rs=data["rs"]
                        )

                        positions.append(asdict(pos))

                        if len(positions) >= self.batch_size:
                            ts = time.strftime("%H%M%S")
                            worker_tag = f"w{self.worker_id}_" if self.worker_id else ""
                            rand_id = uuid.uuid4().hex[:6]
                            with open(self.output_dir / f"hnn_{worker_tag}g{game_id}_b{batch_num}_{ts}_{rand_id}.json", 'w') as f:
                                json.dump({"positions": positions}, f, indent=2)
                            positions, batch_num = [], batch_num + 1

            # Flush any remaining positions at the end of the game (guarantee no lost data)
            if positions:
                ts = time.strftime("%H%M%S")
                worker_tag = f"w{self.worker_id}_" if self.worker_id else ""
                rand_id = uuid.uuid4().hex[:6]
                with open(self.output_dir / f"hnn_{worker_tag}g{game_id}_b{batch_num}_{ts}_{rand_id}.json", 'w') as f:
                    json.dump({"positions": positions}, f, indent=2)

            return total_game_pos

        finally: engine.quit()



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--engine", "-e", default="../nextfish.exe")

    parser.add_argument("--output", "-o", default="D:/nextfish/data")

    parser.add_argument("--games", "-g", type=int, default=100)
    parser.add_argument("--nodes", "-n", type=int, default=16384, help="Stockfish node budget per position (default: 16384)")
    parser.add_argument("--hash", type=int, default=2048, help="Hash table size in MB per engine (default: 2048)")
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Number of positions per JSON file (default: 100)")

    parser.add_argument("--concurrency", "-j", type=int, default=4, help="Number of concurrent worker processes (default: 4)")
    parser.add_argument("--time-budget", type=int, default=None, help="Stop after N seconds (leave time for commit)")
    parser.add_argument("--worker-id", "-w", default=None, help="Worker identifier for naming")

    args = parser.parse_args()

    print(f"=== HARENN Turbo Generator V8 (Raw Data - Budget: {args.nodes:,} Nodes | Hash: {args.hash}MB | Concurrency: {args.concurrency}) ===")

    gen = ParallelGenerator(
        args.engine,
        args.output,
        nodes=args.nodes,
        hash_mb=args.hash,
        time_budget=args.time_budget,
        worker_id=args.worker_id,
        batch_size=args.batch_size
    )

    total_pos, start_time = 0, time.time()

    with ProcessPoolExecutor(max_workers=args.concurrency) as executor:
        futures = set()
        game_id = 0

        # Prime the executor pool
        initial_tasks = min(args.concurrency * 2, args.games)
        for _ in range(initial_tasks):
            futures.add(executor.submit(gen.worker_task, game_id))
            game_id += 1

        while futures:
            time_up = args.time_budget and (time.time() - start_time) >= args.time_budget

            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                res = f.result()
                total_pos += res
                elapsed = time.time() - start_time
                print(f"[{time.strftime('%H:%M:%S')}] Total Progress: {total_pos} Pos | Speed: {total_pos/elapsed:.2f} pos/s", flush=True)

            # Keep feeding tasks if time is not up and more games remain
            while not time_up and len(futures) < args.concurrency * 2 and game_id < args.games:
                futures.add(executor.submit(gen.worker_task, game_id))
                game_id += 1



if __name__ == "__main__": main()

