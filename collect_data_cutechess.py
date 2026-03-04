#!/usr/bin/env python3
"""
HARENN Data Collection using CuteChess

This script generates high-quality training data for HARENN by playing games
and collecting positions with UCI analysis to get the 5 labels.

Usage:
    python collect_data_cutechess.py --engine ./nextfish.exe --games 1000 --output ./data
"""

import argparse
import os
import sys
import json
import time
import random
import subprocess
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import chess
import chess.pgn
import chess.engine


@dataclass
class PositionData:
    """Single position with analysis data"""

    fen: str
    move: str  # Move that was played
    game_result: int  # 0=loss, 1=draw, 2=win
    stm: int  # Side to move
    depth_scores: Dict[int, int]  # depth -> score
    best_move: str
    pv: List[str]
    nodes_searched: int
    time_used: float


class CuteChessCollector:
    """Collect training data using cutechess-cli"""

    def __init__(self, engine_path: str, output_dir: str, tc: str = "10+0.1"):
        self.engine_path = engine_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tc = tc
        self.analysis_engine = None

        # Find cutechess
        self.cutechess = self._find_cutechess()

        # Statistics
        self.stats = {"games_played": 0, "positions_collected": 0, "errors": 0}

    def _find_cutechess(self) -> str:
        """Find cutechess-cli executable"""
        # Check current directory
        for name in ["cutechess-cli.exe", "cutechess-cli", "cutechess"]:
            path = Path(name)
            if path.exists():
                return str(path)

        # Check parent directories
        for parent in [Path("."), Path(".."), Path("../..")]:
            for name in ["cutechess-cli.exe", "cutechess-cli"]:
                path = parent / name
                if path.exists():
                    return str(path)

        raise RuntimeError("cutechess-cli not found")

    def _get_analysis_engine(self) -> chess.engine.SimpleEngine:
        if self.analysis_engine is None:
            self.analysis_engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        return self.analysis_engine

    def _close_analysis_engine(self):
        if self.analysis_engine is not None:
            try:
                self.analysis_engine.quit()
            except Exception:
                pass
            self.analysis_engine = None

    def _analyze_position(self, fen: str, depth: int = 12) -> Dict:
        """Analyze a position using the engine"""

        try:
            engine = self._get_analysis_engine()
            board = chess.Board(fen)
            result = engine.analyse(board, chess.engine.Limit(depth=depth))

            score = result.get(
                "score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)
            )
            cp = score.relative.score(mate_score=10000) or 0

            best_move = (
                result.get("pv", [chess.Move.null()])[0]
                if result.get("pv")
                else chess.Move.null()
            )
            pv = [m.uci() for m in result.get("pv", [])][:10]

            return {"score": cp, "best_move": best_move.uci(), "pv": pv}

        except Exception as e:
            return {"score": 0, "best_move": None, "pv": []}

    def generate_positions_from_games(
        self, num_games: int, output_file: str, games_per_file: int = 100
    ):
        """Generate training data from self-play games"""

        print(f"Generating {num_games} games...")

        all_positions = []

        for game_batch in range(0, num_games, games_per_file):
            batch_size = min(games_per_file, num_games - game_batch)
            print(f"\n--- Games {game_batch + 1} to {game_batch + batch_size} ---")

            # Play games with cutechess
            pgn_file = self.output_dir / f"games_{game_batch}.pgn"

            cmd = [
                self.cutechess,
                "-engine",
                f"cmd={self.engine_path}",
                "name=NF",
                "proto=uci",
                "-engine",
                f"cmd={self.engine_path}",
                "name=SF",
                "proto=uci",
                "-each",
                f"tc={self.tc}",
                "-games",
                str(batch_size),
                "-rounds",
                "1",
                "-pgnout",
                str(pgn_file),
                "-openings",
                "file=../UHO_2022_8mvs_+110_+119.pgn",
                "format=pgn",
                "order=random",
                "plies=8",
                "-repeat",
            ]

            print(f"Running cutechess: {' '.join(cmd[:6])}...")

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.output_dir.parent),
                    capture_output=True,
                    timeout=3600,  # 1 hour timeout
                    text=True,
                )

                # Read generated PGN
                if pgn_file.exists():
                    with open(pgn_file, "r") as f:
                        game_idx = 0
                        while True:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break

                            # Extract positions
                            board = game.board()
                            for move in game.mainline_moves():
                                board.push(move)

                                # Analyze position
                                analysis = self._analyze_position(board.fen(), depth=9)

                                position_data = {
                                    "fen": board.fen(),
                                    "move": move.uci(),
                                    "game_result": self._get_game_result(
                                        game, board.turn
                                    ),
                                    "stm": 1 if board.turn == chess.BLACK else 0,
                                    "score": analysis.get("score", 0),
                                    "best_move": analysis.get("best_move"),
                                    "pv": analysis.get("pv", []),
                                    "depth": 9,
                                }

                                all_positions.append(position_data)
                                self.stats["positions_collected"] += 1

                                # Progress
                                if self.stats["positions_collected"] % 100 == 0:
                                    print(
                                        f"  Positions: {self.stats['positions_collected']}"
                                    )

                            self.stats["games_played"] += 1
                            game_idx += 1

            except subprocess.TimeoutExpired:
                print("Timeout - saving progress...")
            except Exception as e:
                print(f"Error: {e}")
                self.stats["errors"] += 1

            # Save intermediate results
            if len(all_positions) > 0:
                self._save_positions(
                    all_positions[:5000],
                    f"{output_file.split('.')[0]}_{game_batch}.json",
                )

        # Save final results
        self._save_positions(all_positions, output_file)

        self._close_analysis_engine()

        print(f"\n=== Collection Complete ===")
        print(f"Games: {self.stats['games_played']}")
        print(f"Positions: {self.stats['positions_collected']}")
        print(f"Errors: {self.stats['errors']}")

        return all_positions

    def _get_game_result(self, game: chess.pgn.Game, turn: chess.Color) -> int:
        """Get game result from perspective of side to move"""
        result = game.headers.get("Result", "*")

        if result == "1-0":
            return 2 if turn == chess.WHITE else 0
        elif result == "0-1":
            return 0 if turn == chess.WHITE else 2
        else:
            return 1  # Draw

    def _save_positions(self, positions: List[Dict], filename: str):
        """Save positions to JSON file"""
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(
                {
                    "positions": positions,
                    "stats": self.stats,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )

        print(f"Saved {len(positions)} positions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HARENN Data Collection using CuteChess"
    )
    parser.add_argument(
        "--engine", "-e", default="./nextfish.exe", help="Path to chess engine"
    )
    parser.add_argument("--output", "-o", default="./data", help="Output directory")
    parser.add_argument(
        "--games", "-g", type=int, default=100, help="Number of games to play"
    )
    parser.add_argument(
        "--tc", "-t", default="10+0.1", help="Time control (e.g., 10+0.1)"
    )
    parser.add_argument(
        "--batch", "-b", type=int, default=20, help="Games per batch file"
    )

    args = parser.parse_args()

    # Create collector
    collector = CuteChessCollector(
        engine_path=args.engine, output_dir=args.output, tc=args.tc
    )

    # Generate data
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"harenn_positions_{timestamp}.json"

    collector.generate_positions_from_games(
        num_games=args.games, output_file=output_file, games_per_file=args.batch
    )


if __name__ == "__main__":
    main()
