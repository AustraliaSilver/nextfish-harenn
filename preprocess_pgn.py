import argparse
import random
from pathlib import Path

import chess.pgn


def generate_fallback_openings(count: int) -> list[str]:
    first_moves = ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3"]
    responses = {
        "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6", "g8f6"],
        "d2d4": ["d7d5", "g8f6", "e7e6", "c7c5"],
        "c2c4": ["e7e5", "c7c5", "g8f6", "e7e6"],
        "g1f3": ["d7d5", "g8f6", "c7c5", "e7e6"],
        "b1c3": ["d7d5", "g8f6", "e7e5"],
    }
    third_moves = ["g1f3", "b1c3", "f2f4", "d2d4", "e2e4", "c2c4"]

    lines = []
    for _ in range(count):
        m1 = random.choice(first_moves)
        m2 = random.choice(responses[m1])
        m3 = random.choice(third_moves)
        lines.append(f"{m1} {m2} {m3}")
    return lines


def extract_openings_from_pgn(pgn_path: Path, max_lines: int, plies: int) -> list[str]:
    lines: list[str] = []
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
        while len(lines) < max_lines:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            seq = []
            for i, move in enumerate(game.mainline_moves()):
                if i >= plies:
                    break
                seq.append(move.uci())
            if seq:
                lines.append(" ".join(seq))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build opening seeds (UCI moves) for HARENN datagen.")
    parser.add_argument("--input", default="UHO_2022_8mvs_+110_+119.pgn", help="Optional PGN file path")
    parser.add_argument("--output", default="book_moves.txt", help="Output seed line file")
    parser.add_argument("--lines", type=int, default=5000, help="Number of opening seed lines")
    parser.add_argument("--plies", type=int, default=8, help="Maximum plies extracted per game")
    parser.add_argument("--seed", type=int, default=20260224, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    pgn_path = Path(args.input)

    lines: list[str]
    if pgn_path.exists():
        print(f"[book] Extracting from PGN: {pgn_path}")
        lines = extract_openings_from_pgn(pgn_path, args.lines, args.plies)
        if not lines:
            print("[book] PGN parse produced no lines, using fallback random seeds.")
            lines = generate_fallback_openings(args.lines)
    else:
        print(f"[book] PGN not found: {pgn_path}, using fallback random seeds.")
        lines = generate_fallback_openings(args.lines)

    random.shuffle(lines)
    lines = lines[: args.lines]

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[book] Wrote {len(lines)} opening lines to {out}")


if __name__ == "__main__":
    main()