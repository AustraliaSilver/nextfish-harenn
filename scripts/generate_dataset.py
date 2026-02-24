import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ENTRY_STRUCT = struct.Struct('<Q32sbhb4096shh')
ENTRY_SIZE = ENTRY_STRUCT.size  # 4146 bytes


def run(cmd: List[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_binpack(path: Path, sample_limit: int = 4000) -> Dict[str, float]:
    size = path.stat().st_size
    if size % ENTRY_SIZE != 0:
        raise RuntimeError(f"{path} size {size} is not aligned to entry size {ENTRY_SIZE}")

    n = size // ENTRY_SIZE
    if n == 0:
        raise RuntimeError(f"{path} has no entries")

    sampled = min(sample_limit, n)
    step = max(1, n // sampled)

    score_abs_sum = 0
    complexity_sum = 0
    risk_sum = 0
    resolution_sum = 0

    with path.open('rb') as f:
        for i in range(0, n, step):
            if i // step >= sampled:
                break
            f.seek(i * ENTRY_SIZE)
            raw = f.read(ENTRY_SIZE)
            if len(raw) != ENTRY_SIZE:
                break
            _, _, stm, score, result, complexity, _, risk, resolution = ENTRY_STRUCT.unpack(raw)

            if stm not in (-1, 1):
                raise RuntimeError(f"invalid stm={stm} in {path}")
            if result not in (-1, 0, 1):
                raise RuntimeError(f"invalid result={result} in {path}")

            score_abs_sum += abs(score)
            complexity_sum += complexity
            risk_sum += risk
            resolution_sum += resolution

    denom = max(1, sampled)
    return {
        'entries': int(n),
        'avg_abs_score_cp': score_abs_sum / denom,
        'avg_complexity_x100': complexity_sum / denom,
        'avg_risk_x100': risk_sum / denom,
        'avg_resolution_x100': resolution_sum / denom,
    }


def maybe_compress(src: Path, prefer_zstd: bool = True) -> Tuple[Path, str]:
    if prefer_zstd:
        try:
            import zstandard as zstd  # type: ignore

            dst = src.with_suffix(src.suffix + '.zst')
            cctx = zstd.ZstdCompressor(level=10)
            with src.open('rb') as fi, dst.open('wb') as fo:
                cctx.copy_stream(fi, fo)
            return dst, 'zstd'
        except Exception:
            pass

    import gzip

    dst = src.with_suffix(src.suffix + '.gz')
    with src.open('rb') as fi, gzip.open(dst, 'wb', compresslevel=6) as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
    return dst, 'gzip'


def main() -> None:
    p = argparse.ArgumentParser(description='Orchestrate high-quality HARENN dataset generation')
    p.add_argument('--repo-root', default='.')
    p.add_argument('--generator', default='./harenn_gen')
    p.add_argument('--stockfish', default='./stockfish')
    p.add_argument('--games', type=int, default=120)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--book-input', default='books/UHO_2022_8mvs_+110_+119.pgn')
    p.add_argument('--book-lines', type=int, default=8000)
    p.add_argument('--book-plies', type=int, default=8)
    p.add_argument('--out-dir', default='data/generated')
    p.add_argument('--prefix', default='harenn')
    p.add_argument('--keep-shards', action='store_true')
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            'preprocess_pgn.py',
            '--input',
            args.book_input,
            '--output',
            'book_moves.txt',
            '--lines',
            str(args.book_lines),
            '--plies',
            str(args.book_plies),
            '--seed',
            '20260224',
        ],
        cwd=root,
    )

    workers = max(1, args.workers)
    games_per_worker = args.games // workers
    rem = args.games % workers

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    shards: List[Path] = []
    for i in range(workers):
        g = games_per_worker + (1 if i < rem else 0)
        if g <= 0:
            continue
        shard = out_dir / f'{args.prefix}_{stamp}_w{i:02d}.binpack'
        run([args.generator, str(g), str(shard), args.stockfish], cwd=root)
        shards.append(shard)

    if not shards:
        raise RuntimeError('No shards generated')

    stats = {}
    total_entries = 0
    for s in shards:
        st = inspect_binpack(s)
        stats[s.name] = st
        total_entries += int(st['entries'])

    merged = out_dir / f'{args.prefix}_{stamp}.binpack'
    with merged.open('wb') as fo:
        for s in shards:
            with s.open('rb') as fi:
                while True:
                    chunk = fi.read(1 << 20)
                    if not chunk:
                        break
                    fo.write(chunk)

    merged_stats = inspect_binpack(merged)
    compressed, codec = maybe_compress(merged, prefer_zstd=True)

    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'entry_size': ENTRY_SIZE,
        'generator': args.generator,
        'stockfish': args.stockfish,
        'games': args.games,
        'workers': workers,
        'book_input': args.book_input,
        'book_lines': args.book_lines,
        'book_plies': args.book_plies,
        'shards': stats,
        'total_entries': total_entries,
        'merged': {
            'path': str(merged.name),
            'size_bytes': merged.stat().st_size,
            'sha256': sha256_of(merged),
            'stats': merged_stats,
        },
        'compressed': {
            'path': compressed.name,
            'codec': codec,
            'size_bytes': compressed.stat().st_size,
            'sha256': sha256_of(compressed),
        },
    }

    manifest_path = out_dir / f'{args.prefix}_{stamp}.manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    if not args.keep_shards:
        for s in shards:
            try:
                s.unlink()
            except FileNotFoundError:
                pass

    print('[done] merged:', merged)
    print('[done] compressed:', compressed)
    print('[done] manifest:', manifest_path)


if __name__ == '__main__':
    main()
