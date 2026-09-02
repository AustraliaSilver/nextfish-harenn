#!/usr/bin/env python3
"""
Simplified HARENN Training Data Generator

Usage:
    python simple_generate.py --engine ./nextfish.exe --games 100
"""

import argparse
import os
import sys
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass
import chess
import chess.engine


@dataclass
class TrainingPosition:
    fen: str
    stm: int
    eval_score: int
    depth: int
    best_move: str
    game_result: int
    material: int
    piece_count: int
    tau: float


class SimpleDataGenerator:
    def __init__(self, engine_path: str, output_dir: str):
        self.engine_path = engine_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = None

    def get_engine(self):
        """Get or create engine instance"""
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        return self.engine

    def close_engine(self):
        """Close engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None

    def count_material(self, board: chess.Board) -> int:
        """Count material balance"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }
        total = 0
        for piece_type in piece_values:
            total += (
                len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            )
            total -= (
                len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            )
        return total

    def calculate_tau_fast(self, board: chess.Board) -> float:
        """Calculate tactical complexity (τ) - FAST heuristic based on board features"""
        # Heuristic: check + captures + threats = tactical complexity
        try:
            tactical = 0.0

            # Check danger
            if board.is_check():
                tactical += 0.3

            # Piece activity (more pieces = more complex)
            for sq in board.piece_map():
                piece = board.piece_at(sq)
                if piece and piece.piece_type in (
                    chess.QUEEN,
                    chess.ROOK,
                    chess.KNIGHT,
                ):
                    tactical += 0.05

            # King safety (kings near center or under attack)
            wk = board.king(chess.WHITE)
            bk = board.king(chess.BLACK)
            if wk and bk:
                # Kings in center are more vulnerable
                wk_file, wk_rank = chess.square_file(wk), chess.square_rank(wk)
                bk_file, bk_rank = chess.square_file(bk), chess.square_rank(bk)
                if 2 <= wk_file <= 5 and 2 <= wk_rank <= 5:
                    tactical += 0.2
                if 2 <= bk_file <= 5 and 2 <= bk_rank <= 5:
                    tactical += 0.2

            return min(tactical, 1.0)
        except:
            return 0.0

    def analyze_position(self, fen: str, depth: int = 12, engine=None) -> dict:
        """Analyze a position using UCI engine"""

        try:
            # Use passed engine or create one
            if engine is None:
                engine = self.get_engine()

            board = chess.Board(fen)
            limit = chess.engine.Limit(depth=depth)
            result = engine.analyse(board, limit)

            score = result.get("score")
            if score:
                cp = score.relative.score(mate_score=10000)
            else:
                cp = 0

            best_move = (
                result.get("pv", [chess.Move.null()])[0] if result.get("pv") else None
            )

            material = self.count_material(board)
            piece_count = len(board.piece_map())
            tau = self.calculate_tau_fast(board)

            return {
                "score": cp or 0,
                "best_move": best_move.uci() if best_move else "0000",
                "depth": depth,
                "material": material,
                "piece_count": piece_count,
                "tau": tau,
            }
        except Exception as e:
            print(f"Error analyzing: {e}")
            return {
                "score": 0,
                "best_move": "0000",
                "depth": depth,
                "material": 0,
                "piece_count": 0,
                "tau": 0.0,
            }

    def is_quality_position(
        self, board: chess.Board, eval_score: int, tau: float = 0.0
    ) -> bool:
        """Check if position meets quality criteria - FAST checks only"""

        # 1. Not in opening (need at least 6 moves = 12 plies)
        if board.ply() < 12:
            return False

        # 2. Not in endgame (need at least 10 pieces total)
        if len(board.piece_map()) < 10:
            return False

        # 3. Avoid mate or near-mate positions
        if abs(eval_score) > 600:
            return False

        # 4. Avoid positions where one side is winning big
        if abs(eval_score) > 400:
            return False

        # 5. Should have both kings
        if not board.king(chess.WHITE) or not board.king(chess.BLACK):
            return False

        # 6. Should have pawns remaining
        pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + len(
            board.pieces(chess.PAWN, chess.BLACK)
        )
        if pawns < 2:
            return False

        # 7. Prefer more tactical positions (tau > 0.1) OR very balanced (abs(eval) < 100)
        # This keeps interesting positions
        if tau < 0.1 and abs(eval_score) > 100:
            return False

        return True

    def generate_games(self, num_games: int, batch_size: int = 500) -> list:
        """Generate self-play games and collect positions"""

        positions = []
        batch_num = 0

        print(f"Generating {num_games} games...")
        print(f"Saving every {batch_size} positions")

        # Use single engine for all games (faster)
        engine = self.get_engine()

        # Restart engine every 50 games to prevent memory issues
        games_since_restart = 0

        for game_idx in range(num_games):
            try:
                # Restart engine periodically
                if games_since_restart >= 50:
                    self.close_engine()
                    engine = self.get_engine()
                    games_since_restart = 0

                board = chess.Board()

                # Play game
                for move_num in range(100):
                    # Use engine to pick move with some randomness for diversity
                    if random.random() < 0.1:  # 10% random moves for diversity
                        move = random.choice(list(board.legal_moves))
                    else:
                        result = engine.play(board, chess.engine.Limit(time=0.05))
                        move = result.move

                    if move is None:
                        break

                    board.push(move)

                    # Quality-controlled sampling
                    if move_num >= 6 and random.random() < 0.25:
                        # Analyze position (reuse engine)
                        analysis = self.analyze_position(board.fen(), engine=engine)

                        # Apply quality filters (includes τ check)
                        if self.is_quality_position(
                            board, analysis["score"], analysis["tau"]
                        ):
                            pos = TrainingPosition(
                                fen=board.fen(),
                                stm=0 if board.turn == chess.WHITE else 1,
                                eval_score=analysis["score"],
                                depth=analysis["depth"],
                                best_move=analysis["best_move"],
                                game_result=1,
                                material=analysis["material"],
                                piece_count=analysis["piece_count"],
                                tau=analysis["tau"],
                            )
                            positions.append(pos)

                            # Save batch when reaching limit
                            if len(positions) >= batch_size:
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                filename = f"harenn_batch_{batch_num}_{timestamp}.json"
                                self.save_positions(positions, filename)
                                positions = []  # Clear memory
                                batch_num += 1

                    # Check game over
                    if board.is_game_over():
                        break

                games_since_restart += 1

                # Determine game result
                if board.is_checkmate():
                    result = board.result()
                    game_result = {"1-0": 2, "0-1": 0, "1/2-1/2": 1}.get(result, 1)
                elif board.is_game_over():
                    game_result = 1
                else:
                    game_result = 1

                # Update game result for all positions from this game
                for pos in positions[-30:]:
                    pos.game_result = game_result

            except Exception as e:
                print(f"Error in game {game_idx}: {e}")

            if (game_idx + 1) % 10 == 0:
                print(
                    f"  Games: {game_idx + 1}/{num_games}, Positions: {len(positions)}"
                )

        # Close engine
        self.close_engine()

        return positions

    def save_positions(self, positions: list, filename: str):
        """Save positions to JSON file"""

        output_path = self.output_dir / filename

        # Calculate statistics
        eval_scores = [p.eval_score for p in positions]
        tau_values = [p.tau for p in positions]

        data = {
            "positions": [
                {
                    "fen": p.fen,
                    "stm": p.stm,
                    "eval_score": p.eval_score,
                    "depth": p.depth,
                    "best_move": p.best_move,
                    "game_result": p.game_result,
                    "material": p.material,
                    "piece_count": p.piece_count,
                    "tau": round(p.tau, 4),
                }
                for p in positions
            ],
            "stats": {
                "total_positions": len(positions),
                "avg_eval": round(sum(eval_scores) / len(eval_scores), 2)
                if eval_scores
                else 0,
                "avg_tau": round(sum(tau_values) / len(tau_values), 4)
                if tau_values
                else 0,
                "white_wins": sum(1 for p in positions if p.game_result == 2),
                "black_wins": sum(1 for p in positions if p.game_result == 0),
                "draws": sum(1 for p in positions if p.game_result == 1),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(positions)} positions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple HARENN Data Generator")
    parser.add_argument(
        "--engine", "-e", default="./nextfish.exe", help="Path to chess engine"
    )
    parser.add_argument("--output", "-o", default="./data", help="Output directory")
    parser.add_argument("--games", "-g", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Run continuously until Ctrl+C (ignores --games)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        default=500,
        help="Save batch every N positions",
    )

    args = parser.parse_args()

    print(f"Engine: {args.engine}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch}")

    generator = SimpleDataGenerator(args.engine, args.output)

    if args.continuous:
        print("Running CONTINUOUSLY - press Ctrl+C to stop")
        total = 0
        while True:
            try:
                positions = generator.generate_games(1000, batch_size=args.batch)
                total += len(positions)
                print(f"  Total so far: {total} positions")
            except KeyboardInterrupt:
                print(f"\nStopped! Total: {total} positions")
                break
    else:
        print(f"Games: {args.games}")
        positions = generator.generate_games(args.games, batch_size=args.batch)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"harenn_positions_{timestamp}.json"

        generator.save_positions(positions, filename)

        print(f"\nDone! Generated {len(positions)} positions")


if __name__ == "__main__":
    main()
