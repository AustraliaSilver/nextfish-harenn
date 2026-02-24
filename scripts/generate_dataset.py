import argparse
import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ENTRY_STRUCT = struct.Struct("<Q32sbhb4096shh")
ENTRY_SIZE = ENTRY_STRUCT.size  # 4146 bytes


def run(cmd: List[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def popcount_u64(x: int) -> int:
    return x.bit_count()


def unpack_entry(raw: bytes):
    return ENTRY_STRUCT.unpack(raw)


def inspect_binpack(path: Path, sample_limit: int = 8000) -> Dict[str, float]:
    size = path.stat().st_size
    if size % ENTRY_SIZE != 0:
        raise RuntimeError(f"{path} size {size} is not aligned to entry size {ENTRY_SIZE}")

    n = size // ENTRY_SIZE
    if n == 0:
        raise RuntimeError(f"{path} has no entries")

    sampled = min(sample_limit, n)
    step = max(1, n // sampled)

    score_abs_sum = 0.0
    complexity_sum = 0.0
    risk_sum = 0.0
    resolution_sum = 0.0
    mcs_nonzero_sum = 0.0

    phase_opening = 0
    phase_middlegame = 0
    phase_endgame = 0

    with path.open("rb") as f:
        read_count = 0
        for i in range(0, n, step):
            if read_count >= sampled:
                break
            f.seek(i * ENTRY_SIZE)
            raw = f.read(ENTRY_SIZE)
            if len(raw) != ENTRY_SIZE:
                break

            occupancy, _, stm, score, result, complexity, mcs_map, risk, resolution = unpack_entry(raw)

            if stm not in (-1, 1):
                raise RuntimeError(f"invalid stm={stm} in {path}")
            if result not in (-1, 0, 1):
                raise RuntimeError(f"invalid result={result} in {path}")
            if not (0 <= complexity <= 100):
                raise RuntimeError(f"invalid complexity={complexity} in {path}")
            if not (0 <= risk <= 100):
                raise RuntimeError(f"invalid risk={risk} in {path}")
            if not (0 <= resolution <= 100):
                raise RuntimeError(f"invalid resolution={resolution} in {path}")

            pcs = popcount_u64(occupancy)
            if pcs > 24:
                phase_opening += 1
            elif pcs > 12:
                phase_middlegame += 1
            else:
                phase_endgame += 1

            score_abs_sum += abs(score)
            complexity_sum += complexity
            risk_sum += risk
            resolution_sum += resolution
            mcs_nonzero_sum += sum(1 for v in mcs_map if v != 0) / 4096.0
            read_count += 1

    denom = max(1, read_count)
    return {
        "entries": int(n),
        "samples_checked": int(read_count),
        "avg_abs_score_cp": score_abs_sum / denom,
        "avg_complexity_x100": complexity_sum / denom,
        "avg_risk_x100": risk_sum / denom,
        "avg_resolution_x100": resolution_sum / denom,
        "avg_mcs_nonzero_ratio": mcs_nonzero_sum / denom,
        "phase_opening_ratio": phase_opening / denom,
        "phase_middlegame_ratio": phase_middlegame / denom,
        "phase_endgame_ratio": phase_endgame / denom,
    }


def split_dataset(src: Path, out_dir: Path, prefix: str, stamp: str) -> Dict[str, Path]:
    split_paths = {
        "train": out_dir / f"{prefix}_{stamp}.train.binpack",
        "val": out_dir / f"{prefix}_{stamp}.val.binpack",
        "test": out_dir / f"{prefix}_{stamp}.test.binpack",
    }
    files = {k: p.open("wb") for k, p in split_paths.items()}
    counts = {"train": 0, "val": 0, "test": 0}

    with src.open("rb") as fi:
        while True:
            raw = fi.read(ENTRY_SIZE)
            if not raw:
                break
            if len(raw) != ENTRY_SIZE:
                raise RuntimeError(f"truncated entry in {src}")

            # Deterministic split by position key (occupancy+pieces+stm).
            key = raw[:41]
            bucket = hashlib.sha1(key).digest()[0]
            if bucket < 26:  # ~10%
                split = "test"
            elif bucket < 52:  # ~10%
                split = "val"
            else:  # ~80%
                split = "train"

            files[split].write(raw)
            counts[split] += 1

    for f in files.values():
        f.close()

    for split, cnt in counts.items():
        if cnt == 0:
            raise RuntimeError(f"empty split detected: {split}")

    print(f"[split] train={counts['train']} val={counts['val']} test={counts['test']}")
    return split_paths


def maybe_compress(src: Path, prefer_zstd: bool = True) -> Tuple[Path, str]:
    if prefer_zstd:
        try:
            import zstandard as zstd  # type: ignore

            dst = src.with_suffix(src.suffix + ".zst")
            cctx = zstd.ZstdCompressor(level=10)
            with src.open("rb") as fi, dst.open("wb") as fo:
                cctx.copy_stream(fi, fo)
            return dst, "zstd"
        except Exception:
            pass

    import gzip

    dst = src.with_suffix(src.suffix + ".gz")
    with src.open("rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
    return dst, "gzip"


def quality_gate(stats: Dict[str, float], min_entries: int, strict: bool) -> None:
    errors: List[str] = []

    if stats["entries"] < min_entries:
        errors.append(f"entries too low: {stats['entries']} < {min_entries}")

    if not (5.0 <= stats["avg_abs_score_cp"] <= 2500.0):
        errors.append(f"avg_abs_score_cp out of range: {stats['avg_abs_score_cp']:.2f}")

    if not (5.0 <= stats["avg_complexity_x100"] <= 95.0):
        errors.append(f"avg_complexity_x100 out of range: {stats['avg_complexity_x100']:.2f}")

    if not (2.0 <= stats["avg_risk_x100"] <= 98.0):
        errors.append(f"avg_risk_x100 out of range: {stats['avg_risk_x100']:.2f}")

    if not (2.0 <= stats["avg_resolution_x100"] <= 98.0):
        errors.append(f"avg_resolution_x100 out of range: {stats['avg_resolution_x100']:.2f}")

    # Sparse MCS is expected, but all-zero map indicates broken generation.
    if stats["avg_mcs_nonzero_ratio"] < 0.002:
        errors.append(f"avg_mcs_nonzero_ratio too low: {stats['avg_mcs_nonzero_ratio']:.6f}")

    # Require at least two phase buckets represented.
    phase_nonzero = sum(
        1
        for k in ("phase_opening_ratio", "phase_middlegame_ratio", "phase_endgame_ratio")
        if stats[k] > 0.01
    )
    if phase_nonzero < 2:
        errors.append("insufficient phase diversity (need >=2 non-trivial buckets)")

    if errors:
        text = "; ".join(errors)
        if strict:
            raise RuntimeError(f"quality gate failed: {text}")
        print(f"[warn] quality gate warnings: {text}")
    else:
        print("[quality] gate passed")


def main() -> None:
    p = argparse.ArgumentParser(description="Orchestrate high-quality HARENN dataset generation")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--generator", default="./harenn_gen")
    p.add_argument("--stockfish", default="./stockfish")
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--book-input", default="books/UHO_2022_8mvs_+110_+119.pgn")
    p.add_argument("--book-lines", type=int, default=8000)
    p.add_argument("--book-plies", type=int, default=8)
    p.add_argument("--out-dir", default="data/generated")
    p.add_argument("--prefix", default="harenn")
    p.add_argument("--min-entries", type=int, default=2000)
    p.add_argument("--strict-quality", action="store_true")
    p.add_argument("--keep-shards", action="store_true")
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "preprocess_pgn.py",
            "--input",
            args.book_input,
            "--output",
            "book_moves.txt",
            "--lines",
            str(args.book_lines),
            "--plies",
            str(args.book_plies),
            "--seed",
            "20260224",
        ],
        cwd=root,
    )

    workers = max(1, args.workers)
    games_per_worker = args.games // workers
    rem = args.games % workers

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    shards: List[Path] = []
    for i in range(workers):
        g = games_per_worker + (1 if i < rem else 0)
        if g <= 0:
            continue
        shard = out_dir / f"{args.prefix}_{stamp}_w{i:02d}.binpack"
        run([args.generator, str(g), str(shard), args.stockfish], cwd=root)
        shards.append(shard)

    if not shards:
        raise RuntimeError("No shards generated")

    shard_stats = {}
    total_entries = 0
    for s in shards:
        st = inspect_binpack(s)
        shard_stats[s.name] = st
        total_entries += int(st["entries"])

    merged = out_dir / f"{args.prefix}_{stamp}.binpack"
    with merged.open("wb") as fo:
        for s in shards:
            with s.open("rb") as fi:
                while True:
                    chunk = fi.read(1 << 20)
                    if not chunk:
                        break
                    fo.write(chunk)

    merged_stats = inspect_binpack(merged)
    quality_gate(merged_stats, min_entries=args.min_entries, strict=args.strict_quality)

    split_paths = split_dataset(merged, out_dir, args.prefix, stamp)
    split_stats = {name: inspect_binpack(path) for name, path in split_paths.items()}

    compressed, codec = maybe_compress(merged, prefer_zstd=True)
    split_compressed = {}
    for name, path in split_paths.items():
        cp, cc = maybe_compress(path, prefer_zstd=True)
        split_compressed[name] = {
            "path": cp.name,
            "codec": cc,
            "size_bytes": cp.stat().st_size,
            "sha256": sha256_of(cp),
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_size": ENTRY_SIZE,
        "generator": args.generator,
        "stockfish": args.stockfish,
        "games": args.games,
        "workers": workers,
        "book_input": args.book_input,
        "book_lines": args.book_lines,
        "book_plies": args.book_plies,
        "quality": {
            "strict": args.strict_quality,
            "min_entries": args.min_entries,
        },
        "shards": shard_stats,
        "total_entries": total_entries,
        "merged": {
            "path": str(merged.name),
            "size_bytes": merged.stat().st_size,
            "sha256": sha256_of(merged),
            "stats": merged_stats,
        },
        "splits": {
            name: {
                "path": split_paths[name].name,
                "size_bytes": split_paths[name].stat().st_size,
                "sha256": sha256_of(split_paths[name]),
                "stats": split_stats[name],
            }
            for name in split_paths
        },
        "compressed": {
            "path": compressed.name,
            "codec": codec,
            "size_bytes": compressed.stat().st_size,
            "sha256": sha256_of(compressed),
        },
        "split_compressed": split_compressed,
    }

    manifest_path = out_dir / f"{args.prefix}_{stamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.keep_shards:
        for s in shards:
            try:
                s.unlink()
            except FileNotFoundError:
                pass

    print("[done] merged:", merged)
    print("[done] merged compressed:", compressed)
    print("[done] manifest:", manifest_path)


if __name__ == "__main__":
    main()
