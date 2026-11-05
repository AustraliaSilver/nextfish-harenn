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

    wdl: float

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
            # 1. Shallow tier: nodes_shallow = 1024 (Horizon anchor)
            res_shallow = engine.analyse(board, chess.engine.Limit(nodes=self.nodes_shallow))
            m_shallow = res_shallow.get("pv", [None])[0]
            cp_shallow = res_shallow["score"].relative.score(mate_score=10000) or 0

            # 2. Mid tier: nodes_mid = 4096 (Bridge anchor)
            res_mid = engine.analyse(board, chess.engine.Limit(nodes=self.nodes_mid))
            m_mid = res_mid.get("pv", [None])[0]
            cp_mid = res_mid["score"].relative.score(mate_score=10000) or 0

            # 3. Deep tier: nodes = 16384 (MultiPV=2, Exact GM Ground-truth + Move Selection)
            res_deep = engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=2)
            if not isinstance(res_deep, list):
                res_deep = [res_deep]

            m_deep = res_deep[0].get("pv", [None])[0]
            cp_deep = res_deep[0]["score"].relative.score(mate_score=10000) or 0
            cp = cp_deep

            if len(res_deep) >= 2 and res_deep[1].get("pv"):
                m_second = res_deep[1]["pv"][0]
                cp_second = res_deep[1]["score"].relative.score(mate_score=10000) or 0
            else:
                m_second = None
                cp_second = cp_deep

            # 1. RHO (Horizon Risk): Khôi phục 100% công thức gốc max(|1k-4k|, |4k-16k|) / 200 (Zero Drift)
            swing_shallow_mid = abs(cp_shallow - cp_mid)
            swing_mid_deep = abs(cp_mid - cp_deep)
            max_swing = max(swing_shallow_mid, swing_mid_deep)
            rho = min(max_swing / 200.0, 1.0)

            # 2. TAU (Tactical Complexity): Khôi phục 100% 3 mức chuẩn {0.0, 0.5, 1.0} trên 2 bước nhảy
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

            # Bonus cho thế cờ chưa rõ ràng và chiếu vua (chuẩn gốc)
            if abs(cp) < 30:
                rho = min(rho + 0.15, 1.0)
                tau = min(tau + 0.15, 1.0)
            elif abs(cp) < 80:
                rho = min(rho + 0.05, 1.0)

            if board.is_check():
                rho = min(rho + 0.10, 1.0)
                tau = min(tau + 0.10, 1.0)

            # 3. RS (Confidence Score): Khôi phục 100% 4 phân bậc agreement {1.0, 0.75, 0.50, 0.25}
            max_eval_dev = max(abs(cp_shallow - cp_mid), abs(cp_mid - cp_deep), abs(cp_shallow - cp_deep))
            s_stability = max(0.0, 1.0 - (max_eval_dev / 150.0))

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

            margin = max(0, cp_deep - cp_second)
            s_margin = min(1.0, margin / 80.0) if m_second else 0.50

            rs = 0.45 * s_stability + 0.35 * s_agreement + 0.20 * s_margin
            rs = max(0.05, min(0.99, rs))

            m_deep_uci = [m_deep.uci()] if m_deep else []
            l_deep_lbl = [self.move_to_label(board, m_deep)] if m_deep else []
            m_sec_uci = [m_second.uci()] if m_second else m_deep_uci
            l_sec_lbl = [self.move_to_label(board, m_second)] if m_second else l_deep_lbl
            m_mid_uci = [m_mid.uci()] if m_mid else m_deep_uci
            l_mid_lbl = [self.move_to_label(board, m_mid)] if m_mid else l_deep_lbl

            achieved_depth = int(res_deep[0].get("depth", 20))

            return {
                "cp": cp,
                "depth": achieved_depth,
                "best_m": m_deep,
                "sec_m": m_second,
                "cp_second": cp_second,
                "tau": round(tau, 4),
                "rho": round(min(rho, 1.0), 4),
                "rs": round(rs, 4),
                "m16": m_mid_uci,
                "l16": l_mid_lbl,
                "m20": m_sec_uci,
                "l20": l_sec_lbl,
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

        game_positions, board = [], chess.Board()

        try:
            if self.time_budget and (time.time() - self.start_time) > self.time_budget:
                print(f"  [W{worker_id} | G{game_id}] Time budget exceeded, stopping", flush=True)
                return 0

            for move_num in range(1, 100):
                if self.time_budget and (time.time() - self.start_time) > self.time_budget:
                    print(f"  [W{worker_id} | G{game_id}] Time budget reached mid-game, finishing game", flush=True)
                    break

                if board.is_game_over():
                    break

                if move_num <= 6:
                    # Giai đoạn mở cờ nhanh (nước 1-6):
                    # MultiPV=5 ở nodes_shallow: vừa siêu nhanh (~2ms), vừa có sẵn top 5 nước hợp lệ an toàn
                    try:
                        move_info = engine.analyse(board, chess.engine.Limit(nodes=self.nodes_shallow), multipv=min(5, len(list(board.legal_moves))))
                        if not isinstance(move_info, list):
                            move_info = [move_info]

                        best_m = move_info[0]["pv"][0]
                        best_cp = move_info[0]["score"].relative.score(mate_score=10000) or 0

                        # 8% chọn một nước hợp lệ KHÔNG THUA NGAY (eval >= -250 cp) để chống trùng FEN 100%
                        if random.random() < 0.08:
                            safe_cands = [
                                line["pv"][0] for line in move_info
                                if line.get("pv") and (line["score"].relative.score(mate_score=10000) or 0) >= -250
                            ]
                            move = random.choice(safe_cands) if safe_cands else best_m
                        else:
                            # 92% còn lại: Chọn giữa 2 nước tốt nhất (50/50 nếu gap <= 250cp và |sec_cp| <= 500cp)
                            if len(move_info) >= 2 and move_info[1].get("pv"):
                                sec_m = move_info[1]["pv"][0]
                                sec_cp = move_info[1]["score"].relative.score(mate_score=10000) or 0
                                eval_gap = abs(best_cp - sec_cp)
                                if eval_gap <= 250 and abs(sec_cp) <= 500:
                                    move = random.choice([best_m, sec_m])
                                else:
                                    move = best_m
                            else:
                                move = best_m
                    except Exception:
                        move = engine.play(board, chess.engine.Limit(nodes=self.nodes_shallow)).move

                    board.push(move)
                else:
                    # Giai đoạn thu thập dữ liệu (nước > 6):
                    # Tích hợp Analyze-Then-Play: 1 lượt phân tích phục vụ cả gán nhãn chuẩn 3 tầng lẫn đi cờ ở 16,384 nodes!
                    data = self.analyze_full(board, engine)
                    if not data or not data["best_m"]:
                        break

                    chosen_best = data["m24"][0] if data["m24"] else "0000"
                    chosen_best_obj = data["best_m"]
                    pos = TrainingPosition(
                        fen=board.fen(), stm=0 if board.turn == chess.WHITE else 1,
                        eval_score=data["cp"], depth=data.get("depth", 20),
                        best_move=chosen_best,
                        best_move_label=self.move_to_label(board, chosen_best_obj) if chosen_best_obj else 0,
                        best_moves_d16=data["m16"], best_moves_d20=data["m20"], best_moves_d24=data["m24"],
                        best_move_labels_d16=data["l16"], best_move_labels_d20=data["l20"], best_move_labels_d24=data["l24"],
                        game_result=1, wdl=0.5,
                        material=self.count_material(board), piece_count=len(board.piece_map()),
                        tau=data["tau"], rho=data["rho"], rs=data["rs"]
                    )
                    game_positions.append(asdict(pos))

                    # Chọn nước đi tiếp theo trực tiếp từ kết quả 16,384 nodes (không tốn thêm lệnh engine nào!)
                    best_m = data["best_m"]
                    sec_m = data["sec_m"]
                    best_cp = data["cp"]
                    sec_cp = data["cp_second"]

                    if sec_m:
                        eval_gap = abs(best_cp - sec_cp)
                        if move_num <= 8:
                            # 8 nước đầu: 50/50 giữa 2 nước nếu chênh lệch <= 250cp và |eval| <= 500cp
                            if eval_gap <= 250 and abs(sec_cp) <= 500:
                                move = random.choice([best_m, sec_m])
                            else:
                                move = best_m
                        else:
                            # Sau nước 8: 25% chọn nước thứ 2 nếu chênh lệch <= 250cp
                            if eval_gap <= 250 and random.random() < 0.25:
                                move = sec_m
                            else:
                                move = best_m
                    else:
                        move = best_m

                    board.push(move)

            # Xác định kết quả chung cuộc chuẩn xác (Ground-truth Game Result & WDL)
            # white_outcome: 1.0 (Trắng thắng), 0.5 (Hòa), 0.0 (Đen thắng)
            if board.is_checkmate():
                white_outcome = 0.0 if board.turn == chess.WHITE else 1.0
            elif (
                board.is_stalemate()
                or board.is_insufficient_material()
                or board.can_claim_draw()
                or board.is_fivefold_repetition()
                or board.is_seventyfive_moves()
            ):
                white_outcome = 0.5
            else:
                # Trọng tài phán quyết ở nước cuối (Adjudication by Stockfish 19)
                try:
                    final_info = engine.analyse(board, chess.engine.Limit(nodes=4096))
                    final_cp = final_info["score"].white().score(mate_score=10000) or 0
                    if final_cp >= 200:
                        white_outcome = 1.0
                    elif final_cp <= -200:
                        white_outcome = 0.0
                    else:
                        white_outcome = 0.5
                except Exception:
                    white_outcome = 0.5

            # Cập nhật nhãn game_result và wdl theo góc nhìn của bên đi (side to move - stm)
            # stm: 0 (White), 1 (Black)
            for p in game_positions:
                stm = p["stm"]
                stm_win = white_outcome if stm == 0 else (1.0 - white_outcome)
                p["wdl"] = round(stm_win, 2)
                p["game_result"] = 2 if stm_win == 1.0 else (0 if stm_win == 0.0 else 1)

            total_game_pos = len(game_positions)
            if game_positions:
                ts = time.strftime("%H%M%S")
                worker_tag = f"w{self.worker_id}_" if self.worker_id else ""
                rand_id = uuid.uuid4().hex[:6]
                with open(self.output_dir / f"hnn_{worker_tag}g{game_id}_{ts}_{rand_id}.json", 'w') as f:
                    json.dump({"positions": game_positions}, f, separators=(",", ":"))

            outcome_str = "White win" if white_outcome == 1.0 else ("Black win" if white_outcome == 0.0 else "Draw")
            print(f"  [W{worker_id} | G{game_id}] Game finished ({outcome_str}): {total_game_pos} positions collected with Ground-truth WDL", flush=True)
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

