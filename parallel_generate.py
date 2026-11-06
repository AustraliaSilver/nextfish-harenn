#!/usr/bin/env python3

"""

HARENN Turbo Parallel Data Generator - V8 (Raw Data Collection)

D:/nextfish/data - Loại bỏ mọi bộ lọc, phân tích 100% nước đi

"""



import argparse
import os
import sys

# Ensure UTF-8 output encoding across all platforms
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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



    @staticmethod
    def move_to_label(board, move):
        if not move or move == chess.Move.null(): return 0
        piece = board.piece_at(move.from_square)
        if piece is None: return 0
        p_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        return min((move.to_square ^ 56) * 12 + p_idx, 767)

    @staticmethod
    def generate_frc_rank():
        squares = [None] * 8
        # 1. Hai Tượng khác màu ô
        b1 = random.choice([0, 2, 4, 6])
        b2 = random.choice([1, 3, 5, 7])
        squares[b1] = chess.BISHOP
        squares[b2] = chess.BISHOP
        # 2. Đặt Hậu vào 1 ô trống ngẫu nhiên
        empty = [i for i in range(8) if squares[i] is None]
        squares[random.choice(empty)] = chess.QUEEN
        # 3. Đặt 2 Mã vào 2 ô trống ngẫu nhiên
        empty = [i for i in range(8) if squares[i] is None]
        k1, k2 = random.sample(empty, 2)
        squares[k1] = chess.KNIGHT
        squares[k2] = chess.KNIGHT
        # 4. Ba ô còn lại theo thứ tự: Xe, Vua, Xe (Vua luôn ở giữa 2 Xe để đảm bảo nhập thành hợp lệ)
        empty = [i for i in range(8) if squares[i] is None]
        squares[empty[0]] = chess.ROOK
        squares[empty[1]] = chess.KING
        squares[empty[2]] = chess.ROOK
        return squares

    @classmethod
    def generate_dfrc_board(cls):
        """Khởi tạo bàn cờ Double Fischer Random Chess (DFRC - 921,600 thế cờ khởi đầu độc lập)."""
        white_rank = cls.generate_frc_rank()
        black_rank = cls.generate_frc_rank()
        board = chess.Board(None, chess960=True)
        for f in range(8):
            board.set_piece_at(chess.square(f, 0), chess.Piece(white_rank[f], chess.WHITE))
            board.set_piece_at(chess.square(f, 1), chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(chess.square(f, 6), chess.Piece(chess.PAWN, chess.BLACK))
            board.set_piece_at(chess.square(f, 7), chess.Piece(black_rank[f], chess.BLACK))
        board.turn = chess.WHITE
        w_rooks = [f for f in range(8) if white_rank[f] == chess.ROOK]
        b_rooks = [f for f in range(8) if black_rank[f] == chess.ROOK]
        castling_chars = []
        for r in w_rooks:
            castling_chars.append(chess.FILE_NAMES[r].upper())
        for r in b_rooks:
            castling_chars.append(chess.FILE_NAMES[r].lower())
        board.set_castling_fen(''.join(castling_chars))
        return board



    def count_material(self, board):
        w = board.occupied_co[chess.WHITE]
        b = board.occupied_co[chess.BLACK]
        return (
            (chess.popcount(w & board.pawns) - chess.popcount(b & board.pawns)) * 1 +
            (chess.popcount(w & board.knights) - chess.popcount(b & board.knights)) * 3 +
            (chess.popcount(w & board.bishops) - chess.popcount(b & board.bishops)) * 3 +
            (chess.popcount(w & board.rooks) - chess.popcount(b & board.rooks)) * 5 +
            (chess.popcount(w & board.queens) - chess.popcount(b & board.queens)) * 9
        )

    def calculate_rs(self, board, eval_score):
        piece_count = chess.popcount(board.occupied)
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
            # 1. Shallow tier: nodes_shallow = 1024 (MultiPV=2: m_shallow & sec_shallow)
            res_shallow = engine.analyse(board, chess.engine.Limit(nodes=self.nodes_shallow), multipv=2)
            if not isinstance(res_shallow, list):
                res_shallow = [res_shallow]

            m_shallow = res_shallow[0].get("pv", [None])[0]
            sec_shallow = res_shallow[1].get("pv", [None])[0] if len(res_shallow) >= 2 and res_shallow[1].get("pv") else None
            cp_shallow = res_shallow[0]["score"].relative.score(mate_score=10000) or 0

            # 2. Deep tier: nodes = 16384 (MultiPV=2: m_deep & sec_deep, GM Ground-truth + Move Selection)
            res_deep = engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=2)
            if not isinstance(res_deep, list):
                res_deep = [res_deep]

            m_deep = res_deep[0].get("pv", [None])[0]
            sec_deep = res_deep[1].get("pv", [None])[0] if len(res_deep) >= 2 and res_deep[1].get("pv") else None
            cp_deep = res_deep[0]["score"].relative.score(mate_score=10000) or 0
            cp = cp_deep
            cp_second = res_deep[1]["score"].relative.score(mate_score=10000) or 0 if sec_deep else cp_deep

            # 1. RHO (Horizon Risk): |1k - 16k| / 200 (True horizon definition between intuition and deep calculation)
            eval_swing = abs(cp_shallow - cp_deep)
            rho = min(eval_swing / 200.0, 1.0)

            # 2. TAU (Tactical Complexity): 3 levels {0.0, 0.5, 1.0} comparing both best and second-best move pairs
            tau_score = 0.0
            if m_shallow != m_deep:
                tau_score += 0.5
            if sec_shallow is not None and sec_deep is not None and sec_shallow != sec_deep:
                tau_score += 0.5
            tau = tau_score

            # Tactical bonuses for unclear positions and checks
            if abs(cp) < 30:
                rho = min(rho + 0.15, 1.0)
                tau = min(tau + 0.15, 1.0)
            elif abs(cp) < 80:
                rho = min(rho + 0.05, 1.0)

            if board.is_check():
                rho = min(rho + 0.10, 1.0)
                tau = min(tau + 0.10, 1.0)

            # 3. RS (Confidence Score): 4 levels agreement {1.0, 0.75, 0.50, 0.25}
            best_same = (m_shallow == m_deep)
            sec_same = (sec_shallow is not None and sec_deep is not None and sec_shallow == sec_deep)

            if best_same and sec_same:
                s_agreement = 1.0
            elif best_same:
                s_agreement = 0.75
            elif sec_same:
                s_agreement = 0.50
            else:
                s_agreement = 0.25

            s_stability = max(0.0, 1.0 - (eval_swing / 150.0))
            margin = max(0, cp_deep - cp_second)
            s_margin = min(1.0, margin / 80.0) if sec_deep else 0.50

            rs = 0.45 * s_stability + 0.35 * s_agreement + 0.20 * s_margin
            rs = max(0.05, min(0.99, rs))

            m_deep_uci = [m_deep.uci()] if m_deep else []
            l_deep_lbl = [self.move_to_label(board, m_deep)] if m_deep else []
            m_sec_uci = [sec_deep.uci()] if sec_deep else m_deep_uci
            l_sec_lbl = [self.move_to_label(board, sec_deep)] if sec_deep else l_deep_lbl

            achieved_depth = int(res_deep[0].get("depth", 20))

            return {
                "cp": cp,
                "depth": achieved_depth,
                "best_m": m_deep,
                "sec_m": sec_deep,
                "cp_second": cp_second,
                "cp_shallow": cp_shallow,
                "tau": round(tau, 4),
                "rho": round(min(rho, 1.0), 4),
                "rs": round(rs, 4),
                "m16": m_sec_uci,
                "l16": l_sec_lbl,
                "m20": m_deep_uci,
                "l20": l_deep_lbl,
                "m24": m_deep_uci,
                "l24": l_deep_lbl,
            }

        except Exception:
            return None

    def play_one_game(self, engine, game_id, proc_idx):
        game_positions = []
        board = self.generate_dfrc_board()
        last_eval_white = None

        try:
            engine.send_ucinewgame()
        except Exception:
            pass

        for move_num in range(1, 100):
            if self.time_budget and (time.time() - self.start_time) > self.time_budget:
                break

            if board.is_game_over():
                break

            if move_num <= 8:
                # Giai đoạn mở cờ DFRC (nước 1-8):
                # Đi các nước hoàn toàn ngẫu nhiên để xóa bỏ 100% định kiến khai cuộc và tối đa hóa độ đa dạng
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                board.push(move)
            else:
                # Giai đoạn thu thập dữ liệu (nước > 8):
                # Phân tích 16,384 nodes MultiPV=2
                data = self.analyze_full(board, engine)
                if not data or not data["best_m"]:
                    break

                best_m = data["best_m"]
                sec_m = data["sec_m"]
                cp_deep = data["cp"]
                cp_shallow = data.get("cp_shallow", cp_deep)
                last_eval_white = cp_deep if board.turn == chess.WHITE else -cp_deep

                # =========================================================================
                # BỘ LỌC THẾ CỜ TĨNH CHUẨN FINETUNE (QUIET POSITIONS ONLY)
                # =========================================================================
                is_quiet = True

                # 1. Bỏ qua nếu đang bị chiếu hoặc nước đi tốt nhất chiếu tướng đối phương
                if board.is_check() or board.gives_check(best_m):
                    is_quiet = False

                # 2. Bỏ qua nếu nước đi tốt nhất là nước ăn quân (capture)
                if is_quiet and board.is_capture(best_m):
                    is_quiet = False

                # 3. Bỏ qua nếu nước đi tốt nhất là nước phong cấp (promotion)
                if is_quiet and best_m.promotion is not None:
                    is_quiet = False

                # 4. Lọc chênh lệch điểm (Eval-diff filtering):
                # Loại bỏ thế cờ có sự chênh lệch quá lớn giữa đánh giá nông (1k nodes) và sâu (16k nodes)
                # Hoặc biến động tính toán lớn (|cp_shallow - cp_deep| > 150 cp)
                eval_swing = abs(cp_shallow - cp_deep)
                if is_quiet and eval_swing > 150:
                    is_quiet = False

                # 5. Loại bỏ các thế cờ đã ngã ngũ (điểm tuyệt đối >= 1000 centipawns hoặc mate score)
                if is_quiet and abs(cp_deep) >= 1000:
                    is_quiet = False

                # Chỉ lưu lại các thế cờ tĩnh hoàn hảo cho finetuning
                if is_quiet:
                    chosen_best = best_m.uci()
                    chosen_best_lbl = self.move_to_label(board, best_m)
                    sec_best = sec_m.uci() if sec_m else ""
                    sec_best_lbl = self.move_to_label(board, sec_m) if sec_m else 0

                    pos = {
                        "fen": board.fen(),
                        "stm": 0 if board.turn == chess.WHITE else 1,
                        "eval_score": cp_deep,
                        "depth": data.get("depth", 20),
                        "best_move": chosen_best,
                        "best_move_label": chosen_best_lbl,
                        "sec_move": sec_best,
                        "sec_move_label": sec_best_lbl,
                        "best_moves_d16": data["m16"],
                        "best_moves_d20": data["m20"],
                        "best_moves_d24": data["m24"],
                        "best_move_labels_d16": data["l16"],
                        "best_move_labels_d20": data["l20"],
                        "best_move_labels_d24": data["l24"],
                        "game_result": 1,
                        "wdl": 0.5,
                        "material": self.count_material(board),
                        "piece_count": chess.popcount(board.occupied),
                        "tau": data["tau"],
                        "rho": data["rho"],
                        "rs": data["rs"],
                        "is_quiet": True
                    }
                    game_positions.append(pos)

                # Sau nước thứ 8: LUÔN ĐI NƯỚC ĐI TỐT NHẤT (BEST MOVE)
                board.push(best_m)

        # Ground-truth Game Result & WDL
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
        elif last_eval_white is not None:
            if last_eval_white >= 200:
                white_outcome = 1.0
            elif last_eval_white <= -200:
                white_outcome = 0.0
            else:
                white_outcome = 0.5
        else:
            white_outcome = 0.5

        for p in game_positions:
            stm = p["stm"]
            stm_win = white_outcome if stm == 0 else (1.0 - white_outcome)
            p["wdl"] = round(stm_win, 2)
            p["game_result"] = 2 if stm_win == 1.0 else (0 if stm_win == 0.0 else 1)

        outcome_str = "White win" if white_outcome == 1.0 else ("Black win" if white_outcome == 0.0 else "Draw")
        return game_positions, outcome_str

    def worker_loop(self, proc_idx, max_games_per_proc, shared_pos):
        # Ghim chặt worker vào CPU core riêng biệt trên Linux/Cloud runner
        try:
            if hasattr(os, "sched_setaffinity") and hasattr(os, "cpu_count"):
                cpu_total = os.cpu_count() or 4
                os.sched_setaffinity(0, {proc_idx % cpu_total})
        except Exception:
            pass

        ts = time.strftime("%Y%m%d_%H%M%S")
        worker_tag = f"w{self.worker_id}_" if self.worker_id else ""
        jsonl_filename = f"hnn_{worker_tag}p{proc_idx}_{ts}.jsonl"
        jsonl_path = self.output_dir / jsonl_filename

        print(f"  [W{self.worker_id}-P{proc_idx}] Booting persistent Stockfish (Hash={self.hash_mb}MB)...", flush=True)
        engine = chess.engine.SimpleEngine.popen_uci(self.engine_path, timeout=60)
        try:
            engine.configure({"Threads": 1, "Hash": self.hash_mb, "Move Overhead": 0})
        except Exception:
            pass

        games_played = 0
        total_proc_pos = 0

        try:
            with open(jsonl_path, "a", encoding="utf-8") as out_f:
                while games_played < max_games_per_proc:
                    if self.time_budget and (time.time() - self.start_time) >= self.time_budget:
                        print(f"  [W{self.worker_id}-P{proc_idx}] Time budget reached, stopping gracefully.", flush=True)
                        break

                    games_played += 1
                    game_positions, outcome_str = self.play_one_game(engine, games_played, proc_idx)

                    # Ghi trực tiếp nối đuôi vào duy nhất 1 file JSONL (0 file rác, crash-safe)
                    if game_positions:
                        for p in game_positions:
                            out_f.write(json.dumps(p) + "\n")
                        out_f.flush()

                    pos_count = len(game_positions)
                    total_proc_pos += pos_count
                    with shared_pos.get_lock():
                        shared_pos.value += pos_count
                        cur_total = shared_pos.value

                    elapsed = time.time() - self.start_time
                    print(f"[{time.strftime('%H:%M:%S')}] [W{self.worker_id}-P{proc_idx} | G{games_played}] ({outcome_str}) +{pos_count} pos -> {jsonl_filename} | Total: {cur_total:,} Pos ({cur_total/elapsed:.2f} pos/s)", flush=True)

        finally:
            print(f"  [W{self.worker_id}-P{proc_idx}] Shutting down persistent engine after {games_played} games.", flush=True)
            try:
                engine.quit()
            except Exception:
                pass


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

    print(f"=== HARENN Turbo Generator V9 (Persistent Engines + Direct JSONL Streaming | Budget: {args.nodes:,} Nodes | Hash: {args.hash}MB | Concurrency: {args.concurrency}) ===")

    gen = ParallelGenerator(
        args.engine,
        args.output,
        nodes=args.nodes,
        hash_mb=args.hash,
        time_budget=args.time_budget,
        worker_id=args.worker_id,
        batch_size=args.batch_size
    )

    shared_pos = multiprocessing.Value('i', 0)
    if args.games <= 0:
        games_per_proc = float('inf')
    else:
        games_per_proc = max(1, args.games // args.concurrency)

    processes = []
    for proc_idx in range(args.concurrency):
        p = multiprocessing.Process(
            target=gen.worker_loop,
            args=(proc_idx, games_per_proc, shared_pos)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    total_time = time.time() - gen.start_time
    print(f"=== Generation Run Completed! Total: {shared_pos.value:,} positions in {total_time:.1f}s ({shared_pos.value/total_time:.2f} pos/s) ===")


if __name__ == "__main__": main()

