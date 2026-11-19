#!/usr/bin/env python3
"""
HARENN HalfKPv3 - Kaggle Auto HF-to-NPZ Packer & Kaggle Dataset Publisher
========================================================================
Chạy tự động trên Kaggle (hoặc máy local):
  1. Đọc danh sách các file .jsonl/.json đã xử lý từ Kaggle Index Dataset (tránh làm trùng).
  2. Liệt kê các file mới trên Hugging Face Bucket (mặc định: australia1/myrandombucket).
  3. Tải theo từng cụm (--files-per-shard), giải mã đa nhân CPU sang mảng HalfKPv3 nén (.npz):
       - us_indices (uint16[N, 32]), them_indices (uint16[N, 32])
       - labels_eval (float32[N]), labels_wdl (float32[N]), labels_eval_mask (uint8[N])
       - labels_tau, labels_rho, labels_rs (float32[N])
       - best_move_label, sec_move_label (uint16[N])
     Xóa ngay file .jsonl thô sau khi nén xong để không đầy ổ đĩa Kaggle.
  4. Khi gom đủ số shard (--shards-per-dataset) hoặc hết file mới, tự động đăng lên Kaggle Dataset
     dạng part đánh số (p001, p002, ...) và cập nhật Index Dataset để phiên sau tiếp tục nối tiếp.

Ví dụ chạy trên Kaggle Notebook:
  !curl -sL https://raw.githubusercontent.com/AustraliaSilver/nextfish-harenn/data/kaggle_pack_npz_dataset.py -o kaggle_pack_npz_dataset.py
  !python3 kaggle_pack_npz_dataset.py --files-per-shard 17 --shards-per-dataset 4
"""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import orjson as _orjson
    def _fast_json_loads(raw):
        return _orjson.loads(raw)
except ImportError:
    def _fast_json_loads(raw):
        return json.loads(raw)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─── Hằng số HalfKPv3 khớp bit-exact với train_halfkpv3.py & C++ ───
HKV3_KING_BUCKETS = 32
HKV3_CHANNELS = 11
HKV3_PLANES_PER_BUCKET = HKV3_CHANNELS * 64  # 704
HKV3_NUM_FEATURES = HKV3_KING_BUCKETS * HKV3_PLANES_PER_BUCKET  # 22,528
MAX_ACTIVE_FEATURES = 32
PADDING_FEATURE_IDX = HKV3_NUM_FEATURES  # 22,528
WDL_SCALE_PAWNS = 2.60

P2CH_WHITE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4,
    'p': 5, 'n': 6, 'b': 7, 'r': 8, 'q': 9, 'k': 10
}
P2CH_BLACK = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4,
    'P': 5, 'N': 6, 'B': 7, 'R': 8, 'Q': 9, 'K': 10
}


def get_secret(name: str, default: str = "") -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        from kaggle_secrets import UserSecretsClient
        val = UserSecretsClient().get_secret(name) or ""
        if val:
            os.environ[name] = val
            return val
    except Exception:
        pass
    return default


def fen_to_hkv3_sparse_pair(fen: str, stm_override: Optional[object] = None) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    parts = fen.split()
    if not parts:
        return None
    rows = parts[0].split('/')
    if len(rows) != 8:
        return None

    if stm_override is not None:
        if isinstance(stm_override, str):
            s_low = stm_override.strip().lower()
            if s_low in ("b", "black", "1", "true"):
                stm = 1
            elif s_low in ("w", "white", "0", "false"):
                stm = 0
            else:
                stm = 1 if (len(parts) > 1 and parts[1] == 'b') else 0
        else:
            try:
                stm = 1 if int(stm_override) != 0 else 0
            except (ValueError, TypeError):
                stm = 1 if (len(parts) > 1 and parts[1] == 'b') else 0
    else:
        stm = 1 if (len(parts) > 1 and parts[1] == 'b') else 0

    w_ksq = -1
    b_ksq = -1
    pieces = []
    for r_idx, row in enumerate(rows):
        rank = 7 - r_idx
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                sq = rank * 8 + file
                if ch == 'K':
                    w_ksq = sq
                elif ch == 'k':
                    b_ksq = sq
                pieces.append((sq, ch))
                file += 1

    if w_ksq < 0 or b_ksq < 0:
        return None

    w_mirror = 7 if (w_ksq & 7) >= 4 else 0
    w_knorm = w_ksq ^ w_mirror
    w_kb = ((w_knorm >> 3) * 4 + (w_knorm & 3)) * HKV3_PLANES_PER_BUCKET

    b_korient = b_ksq ^ 56
    b_mirror = 7 if (b_korient & 7) >= 4 else 0
    b_knorm = b_korient ^ b_mirror
    b_kb = ((b_knorm >> 3) * 4 + (b_knorm & 3)) * HKV3_PLANES_PER_BUCKET

    w_arr = np.full(MAX_ACTIVE_FEATURES, PADDING_FEATURE_IDX, dtype=np.uint16)
    b_arr = np.full(MAX_ACTIVE_FEATURES, PADDING_FEATURE_IDX, dtype=np.uint16)
    w_cnt = 0
    b_cnt = 0

    for sq, ch in pieces:
        if ch != 'K' and w_cnt < MAX_ACTIVE_FEATURES:
            ch_w = P2CH_WHITE.get(ch)
            if ch_w is not None:
                w_arr[w_cnt] = w_kb + ch_w * 64 + (sq ^ w_mirror)
                w_cnt += 1
        if ch != 'k' and b_cnt < MAX_ACTIVE_FEATURES:
            ch_b = P2CH_BLACK.get(ch)
            if ch_b is not None:
                b_arr[b_cnt] = b_kb + ch_b * 64 + ((sq ^ 56) ^ b_mirror)
                b_cnt += 1

    if stm == 0:
        return w_arr, b_arr, 0
    return b_arr, w_arr, 1


_KNIGHT_DIRS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
_KING_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _is_sq_attacked(grid, sq: int, by_white: bool) -> bool:
    r, f = sq >> 3, sq & 7
    p_atk = "P" if by_white else "p"
    n_atk = "N" if by_white else "n"
    k_atk = "K" if by_white else "k"
    rq_atk = ("R", "Q") if by_white else ("r", "q")
    bq_atk = ("B", "Q") if by_white else ("b", "q")

    # Pawn attacks: white pawn attacks from r-1, black pawn attacks from r+1
    pr = r - 1 if by_white else r + 1
    if 0 <= pr < 8:
        base = pr << 3
        if f > 0 and grid[base + f - 1] == p_atk:
            return True
        if f < 7 and grid[base + f + 1] == p_atk:
            return True

    for dr, df in _KNIGHT_DIRS:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8 and grid[(nr << 3) + nf] == n_atk:
            return True

    for dr, df in _KING_DIRS:
        nr, nf = r + dr, f + df
        if 0 <= nr < 8 and 0 <= nf < 8 and grid[(nr << 3) + nf] == k_atk:
            return True

    for dr, df in _ORTHO_DIRS:
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            pc = grid[(nr << 3) + nf]
            if pc is not None:
                if pc in rq_atk:
                    return True
                break
            nr += dr
            nf += df

    for dr, df in _DIAG_DIRS:
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            pc = grid[(nr << 3) + nf]
            if pc is not None:
                if pc in bq_atk:
                    return True
                break
            nr += dr
            nf += df

    return False


def _infer_is_quiet(board_fen: str, turn_char: str, best_move_uci: str, ev_cp_raw: float, rho: float) -> bool:
    if abs(ev_cp_raw) >= 1000.0 or rho >= 0.75:
        return False

    grid = [None] * 64
    wk = bk = -1
    r, f = 7, 0
    for ch in board_fen:
        if ch == "/":
            r -= 1
            f = 0
        elif "1" <= ch <= "8":
            f += ord(ch) - 48
        else:
            sq = (r << 3) + f
            grid[sq] = ch
            if ch == "K":
                wk = sq
            elif ch == "k":
                bk = sq
            f += 1

    if wk < 0 or bk < 0:
        return False

    us_white = (turn_char == "w")
    our_king = wk if us_white else bk
    opp_king = bk if us_white else wk

    # 1. Currently in check
    if _is_sq_attacked(grid, our_king, not us_white):
        return False

    # 2. Best move is capture, promotion, en-passant, or gives check
    if best_move_uci and len(best_move_uci) >= 4:
        if len(best_move_uci) >= 5:
            return False  # promotion
        f1 = ord(best_move_uci[0]) - 97
        r1 = ord(best_move_uci[1]) - 49
        f2 = ord(best_move_uci[2]) - 97
        r2 = ord(best_move_uci[3]) - 49
        if 0 <= f1 < 8 and 0 <= r1 < 8 and 0 <= f2 < 8 and 0 <= r2 < 8:
            sq1 = (r1 << 3) + f1
            sq2 = (r2 << 3) + f2
            moving = grid[sq1]
            target = grid[sq2]
            if target is not None:
                # DFRC castling encodes king capturing own rook -> check if same color rook
                if not ((moving == "K" and target == "R") or (moving == "k" and target == "r")):
                    return False
            if moving in ("P", "p") and f1 != f2:
                return False  # en passant capture

            # Simulate move to test gives_check
            grid[sq1] = None
            grid[sq2] = moving
            if _is_sq_attacked(grid, opp_king, us_white):
                return False

    return True


def _parse_file_worker(file_path_str: str):
    path = Path(file_path_str)
    hashes, us_l, them_l = [], [], []
    eval_l, wdl_l, mask_l = [], [], []
    tau_l, rho_l, rs_l = [], [], []
    bm_l, sm_l = [], []

    def process_record(pos: dict):
        if not isinstance(pos, dict):
            return
        fen = pos.get("fen") or pos.get("FEN") or pos.get("fen_string") or pos.get("board")
        if not fen or not isinstance(fen, str):
            return
        parts = fen.split()
        if len(parts) < 2:
            return

        board_key = " ".join(parts[:4]) if len(parts) >= 4 else fen
        h = struct.unpack("<q", hashlib.blake2b(board_key.encode("utf-8"), digest_size=8).digest())[0]

        stm = None
        for k in ("stm", "turn", "side_to_move", "to_move"):
            if k in pos and pos[k] is not None:
                stm = pos[k]
                break
        parsed = fen_to_hkv3_sparse_pair(fen, stm)
        if parsed is None:
            return
        us_arr, them_arr, _ = parsed

        ev_raw = None
        for k in ("eval_score", "eval", "score", "cp", "evaluation"):
            if k in pos:
                ev_raw = pos[k]
                break
        if ev_raw is None:
            return
        try:
            ev_cp_raw = float(ev_raw)
        except (ValueError, TypeError):
            return

        tau = max(0.0, min(1.0, float(pos.get("tau", 0.5))))
        rho = max(0.0, min(1.0, float(pos.get("rho", 0.3))))
        rs = max(0.0, min(1.0, float(pos.get("rs", 0.5))))
        bm_lbl = max(0, min(767, int(pos.get("best_move_label", 0))))
        sm_lbl = max(0, min(767, int(pos.get("sec_move_label", 0))))

        if "is_quiet" in pos and pos["is_quiet"] is not None:
            q_raw = pos["is_quiet"]
            if isinstance(q_raw, str):
                is_quiet = q_raw.strip().lower() in ("true", "1", "yes")
            else:
                is_quiet = bool(q_raw)
        else:
            bm_uci = pos.get("best_move") or ""
            is_quiet = _infer_is_quiet(parts[0], parts[1].lower(), bm_uci, ev_cp_raw, rho)
        eval_mask = 1 if (is_quiet and abs(ev_cp_raw) < 9000.0) else 0

        ev_pawns = max(-1000.0, min(1000.0, ev_cp_raw)) / 100.0

        if "wdl" in pos and pos["wdl"] is not None:
            try:
                wdl_val = max(0.0, min(1.0, float(pos["wdl"])))
            except (ValueError, TypeError):
                wdl_val = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, ev_pawns / WDL_SCALE_PAWNS))))
        elif "game_result" in pos and pos["game_result"] is not None:
            try:
                wdl_val = max(0.0, min(1.0, float(pos["game_result"]) / 2.0))
            except (ValueError, TypeError):
                wdl_val = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, ev_pawns / WDL_SCALE_PAWNS))))
        else:
            wdl_val = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, ev_pawns / WDL_SCALE_PAWNS))))

        hashes.append(h)
        us_l.append(us_arr)
        them_l.append(them_arr)
        eval_l.append(ev_pawns)
        wdl_l.append(wdl_val)
        mask_l.append(eval_mask)
        tau_l.append(tau)
        rho_l.append(rho)
        rs_l.append(rs)
        bm_l.append(bm_lbl)
        sm_l.append(sm_lbl)

    try:
        if path.suffix == ".json":
            with open(path, "rb") as f:
                data = _fast_json_loads(f.read())
            positions = data.get("positions", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for p in positions:
                process_record(p)
        else:
            with open(path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        process_record(_fast_json_loads(line))
                    except Exception:
                        continue
    except Exception as e:
        print(f"[Worker Warning] {path.name}: {e}", flush=True)

    if not eval_l:
        return None

    return (
        np.asarray(hashes, dtype=np.int64),
        np.vstack(us_l).astype(np.uint16, copy=False),
        np.vstack(them_l).astype(np.uint16, copy=False),
        np.asarray(eval_l, dtype=np.float32),
        np.asarray(wdl_l, dtype=np.float32),
        np.asarray(mask_l, dtype=np.uint8),
        np.asarray(tau_l, dtype=np.float32),
        np.asarray(rho_l, dtype=np.float32),
        np.asarray(rs_l, dtype=np.float32),
        np.asarray(bm_l, dtype=np.uint16),
        np.asarray(sm_l, dtype=np.uint16),
    )


_FAST_PACK_BIN: Optional[str] = None


def ensure_fast_pack(work_dir: Path) -> Optional[str]:
    global _FAST_PACK_BIN
    if _FAST_PACK_BIN is not None:
        return _FAST_PACK_BIN or None
    exe_name = "fast_pack.exe" if os.name == "nt" else "fast_pack"
    bin_path = work_dir / exe_name
    if bin_path.exists():
        _FAST_PACK_BIN = str(bin_path)
        return _FAST_PACK_BIN
    cpp_path = Path(__file__).with_name("fast_pack.cpp")
    if not cpp_path.exists():
        cpp_dl = work_dir / "fast_pack.cpp"
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/AustraliaSilver/nextfish-harenn/data/fast_pack.cpp",
                str(cpp_dl),
            )
            cpp_path = cpp_dl
        except Exception as e:
            print(f"[FastPack] Khong tai duoc fast_pack.cpp ({e}), dung Python fallback.", flush=True)
            _FAST_PACK_BIN = ""
            return None
    try:
        cmd = ["g++", "-O3", "-std=c++17", "-pthread", str(cpp_path), "-o", str(bin_path)]
        if os.name == "nt":
            cmd = ["g++", "-O3", "-std=c++17", str(cpp_path), "-o", str(bin_path)]
        subprocess.run(cmd, check=True)
        _FAST_PACK_BIN = str(bin_path)
        print(f"[FastPack] Da bien dich C++ fast_pack thanh cong: {_FAST_PACK_BIN}", flush=True)
        return _FAST_PACK_BIN
    except Exception as e:
        print(f"[FastPack] Bien dich that bai ({e}), dung Python fallback.", flush=True)
        _FAST_PACK_BIN = ""
        return None


def _pack_npy_dir_to_npz(npy_dir: Path, out_npz_path: Path) -> None:
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(out_npz_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for p in sorted(npy_dir.glob("*.npy")):
            zf.write(str(p), arcname=p.name)


def build_npz_shard(file_paths: List[str], out_npz_path: Path, dedup: bool = True, work_dir: Optional[Path] = None) -> Tuple[int, int]:
    fp_bin = ensure_fast_pack(work_dir) if work_dir is not None else None
    if fp_bin and all(p.endswith(".jsonl") for p in file_paths):
        npy_dir = (work_dir / "npy_tmp")
        shutil.rmtree(npy_dir, ignore_errors=True)
        npy_dir.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                [fp_bin, str(npy_dir), "1" if dedup else "0", *file_paths],
                capture_output=True, text=True, check=True,
            )
            parts = res.stdout.strip().split()
            raw_n, uniq_n = int(parts[0]), int(parts[1])
            if raw_n == 0 or uniq_n == 0:
                return 0, 0
            _pack_npy_dir_to_npz(npy_dir, out_npz_path)
            return raw_n, uniq_n
        except Exception as e:
            print(f"[FastPack Warning] Loi C++ ({e}), chuyen sang Python fallback...", flush=True)
        finally:
            shutil.rmtree(npy_dir, ignore_errors=True)

    workers = max(1, min(8, cpu_count()))
    with Pool(workers) as pool:
        results = pool.map(_parse_file_worker, file_paths)
    valid = [r for r in results if r is not None]
    del results
    if not valid:
        return 0, 0

    all_hashes = np.concatenate([r[0] for r in valid], axis=0)
    all_us = np.concatenate([r[1] for r in valid], axis=0)
    all_them = np.concatenate([r[2] for r in valid], axis=0)
    all_eval = np.concatenate([r[3] for r in valid], axis=0)
    all_wdl = np.concatenate([r[4] for r in valid], axis=0)
    all_mask = np.concatenate([r[5] for r in valid], axis=0)
    all_tau = np.concatenate([r[6] for r in valid], axis=0)
    all_rho = np.concatenate([r[7] for r in valid], axis=0)
    all_rs = np.concatenate([r[8] for r in valid], axis=0)
    all_bm = np.concatenate([r[9] for r in valid], axis=0)
    all_sm = np.concatenate([r[10] for r in valid], axis=0)
    del valid
    gc.collect()

    raw_n = len(all_eval)
    if dedup:
        _, unique_idx, inv_idx, counts = np.unique(
            all_hashes, return_index=True, return_inverse=True, return_counts=True
        )
        counts_f = counts.astype(np.float32)
        us_out = all_us[unique_idx]
        them_out = all_them[unique_idx]
        bm_out = all_bm[unique_idx]
        sm_out = all_sm[unique_idx]
        eval_out = (np.bincount(inv_idx, weights=all_eval) / counts_f).astype(np.float32)
        wdl_out = (np.bincount(inv_idx, weights=all_wdl) / counts_f).astype(np.float32)
        mask_out = ((np.bincount(inv_idx, weights=all_mask.astype(np.float32)) / counts_f) >= 0.5).astype(np.uint8)
        tau_out = (np.bincount(inv_idx, weights=all_tau) / counts_f).astype(np.float32)
        rho_out = (np.bincount(inv_idx, weights=all_rho) / counts_f).astype(np.float32)
        rs_out = (np.bincount(inv_idx, weights=all_rs) / counts_f).astype(np.float32)
        del unique_idx, inv_idx, counts, counts_f
    else:
        us_out, them_out = all_us, all_them
        eval_out, wdl_out, mask_out = all_eval, all_wdl, all_mask
        tau_out, rho_out, rs_out = all_tau, all_rho, all_rs
        bm_out, sm_out = all_bm, all_sm

    del all_hashes, all_us, all_them, all_eval, all_wdl, all_mask, all_tau, all_rho, all_rs, all_bm, all_sm
    gc.collect()

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_npz_path),
        us_indices=us_out,
        them_indices=them_out,
        labels_eval=eval_out,
        labels_wdl=wdl_out,
        labels_eval_mask=mask_out,
        labels_tau=tau_out,
        labels_rho=rho_out,
        labels_rs=rs_out,
        best_move_label=bm_out,
        sec_move_label=sm_out,
    )
    return raw_n, len(eval_out)


def init_kaggle_api():
    token = get_secret("KAGGLE_API_TOKEN")
    user = get_secret("KAGGLE_USERNAME", "hunganh12333")
    key = get_secret("KAGGLE_KEY")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    if user and key:
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api, user


def load_manifest(k_api, index_dataset_ref: str, work_dir: Path) -> Dict:
    idx_dir = work_dir / "_index_cache"
    idx_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = idx_dir / "packed_manifest.json"
    try:
        k_api.dataset_download_file(index_dataset_ref, "packed_manifest.json", path=str(idx_dir), force=True)
        # Kaggle may save .zip if compressed
        zip_p = idx_dir / "packed_manifest.json.zip"
        if zip_p.exists():
            import zipfile
            with zipfile.ZipFile(zip_p, "r") as zf:
                zf.extractall(idx_dir)
            zip_p.unlink(missing_ok=True)
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[Manifest] Da tai index tu {index_dataset_ref}: {len(data.get('processed_files', []))} file da xu ly, part tiep theo = {data.get('next_part', 1)}", flush=True)
            return data
    except Exception as e:
        print(f"[Manifest] Chua co index dataset '{index_dataset_ref}' ({e}) -> khoi tao manifest moi.", flush=True)
    return {"processed_files": [], "parts": [], "next_part": 1}


def save_and_push_manifest(k_api, index_dataset_ref: str, manifest: Dict, work_dir: Path):
    owner, slug = index_dataset_ref.split("/", 1)
    idx_dir = work_dir / "_index_upload"
    if idx_dir.exists():
        shutil.rmtree(idx_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)

    with open(idx_dir / "packed_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    meta = {
        "title": "HARENN HKV3 NPZ Manifest Index",
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(idx_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    try:
        k_api.dataset_status(index_dataset_ref)
        k_api.dataset_create_version(str(idx_dir), version_notes=f"Update manifest: {len(manifest['processed_files'])} files", dir_mode="zip", quiet=False)
    except Exception:
        k_api.dataset_create_new(str(idx_dir), public=False, dir_mode="zip", quiet=False)
    shutil.rmtree(idx_dir, ignore_errors=True)


def publish_shards_to_kaggle(k_api, owner: str, base_slug: str, part_num: int, shard_paths: List[Path], is_public: bool):
    slug = f"{base_slug}-p{part_num:03d}"
    ds_ref = f"{owner}/{slug}"
    stage_dir = shard_paths[0].parent / f"_stage_p{part_num:03d}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    total_mb = 0.0
    for sp in shard_paths:
        shutil.move(str(sp), str(stage_dir / sp.name))
        total_mb += (stage_dir / sp.name).stat().st_size / (1024 * 1024)

    meta = {
        "title": f"HARENN HKV3 NPZ Dataset Part {part_num:03d}",
        "id": ds_ref,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(stage_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Kaggle Upload] Dang day {len(shard_paths)} shards ({total_mb:.1f} MB) len {ds_ref}...", flush=True)
    try:
        k_api.dataset_status(ds_ref)
        k_api.dataset_create_version(
            str(stage_dir),
            version_notes=f"Auto-packed {len(shard_paths)} HKV3 NPZ shards ({total_mb:.1f} MB)",
            dir_mode="zip",
            quiet=False,
        )
    except Exception:
        k_api.dataset_create_new(str(stage_dir), public=is_public, dir_mode="zip", quiet=False)

    shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"[Kaggle Upload] Hoan tat -> https://www.kaggle.com/datasets/{ds_ref}", flush=True)
    return ds_ref


def main():
    parser = argparse.ArgumentParser(description="Kaggle Auto HF JSONL -> Compressed HKV3 NPZ Dataset Publisher")
    parser.add_argument("--hf-bucket", type=str, default="australia1/myrandombucket", help="Hugging Face Bucket ID")
    parser.add_argument("--kaggle-owner", type=str, default="", help="Kaggle username (mac dinh lay tu env/secret)")
    parser.add_argument("--dataset-prefix", type=str, default="harenn-hkv3-npz", help="Ten goc Kaggle Dataset (tao dang <prefix>-p001, p002...)")
    parser.add_argument("--index-dataset", type=str, default="harenn-hkv3-npz-index", help="Slug dataset nho luu packed_manifest.json")
    parser.add_argument("--files-per-shard", type=int, default=17, help="So file .jsonl toi da gom thanh 1 shard .npz (mac dinh: 17)")
    parser.add_argument("--max-batch-gb", type=float, default=8.0, help="Gioi han dung luong raw JSONL tai moi luot (GB, mac dinh 8.0 de tranh tran o dia Kaggle)")
    parser.add_argument("--shards-per-dataset", type=int, default=4, help="So shard .npz gom du roi tu dong dang len 1 Kaggle Dataset (mac dinh: 4)")
    parser.add_argument("--max-files", type=int, default=0, help="Gioi han so file xu ly moi phien (0 = khong gioi han)")
    parser.add_argument("--time-budget", type=int, default=39600, help="Gioi han thoi gian chay (giay, mac dinh 11h cho Kaggle)")
    parser.add_argument("--work-dir", type=str, default="/kaggle/working/packer_tmp" if os.path.exists("/kaggle/working") else "./packer_tmp", help="Thu muc tam")
    parser.add_argument("--public", action="store_true", help="Tao Kaggle Dataset dang public (mac dinh private)")
    parser.add_argument("--no-dedup", action="store_true", help="Tat khử trùng trong moi shard")
    parser.add_argument("--dry-run", action="store_true", help="Chi liet ke & nen thu, khong day len Kaggle")
    args = parser.parse_args()

    t_start = time.time()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "raw_jsonl"
    shard_dir = work_dir / "shards_npz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    hf_token = get_secret("HF_TOKEN")
    from huggingface_hub import HfApi
    hf_api = HfApi(token=hf_token if hf_token else None)

    if not args.dry_run:
        k_api, detected_user = init_kaggle_api()
        owner = args.kaggle_owner or detected_user or "hunganh12333"
        index_ref = f"{owner}/{args.index_dataset}" if "/" not in args.index_dataset else args.index_dataset
        manifest = load_manifest(k_api, index_ref, work_dir)
    else:
        k_api, owner, index_ref = None, args.kaggle_owner or "hunganh12333", ""
        manifest = {"processed_files": [], "parts": [], "next_part": 1}

    done_set = set(manifest.get("processed_files", []))

    print(f"[HF] Dang liet ke file tren '{args.hf_bucket}'...", flush=True)
    remote_files = sorted(
        [
            (getattr(f, "path", ""), getattr(f, "size", 0) or 0)
            for f in hf_api.list_bucket_tree(args.hf_bucket)
            if (getattr(f, "path", "").endswith(".jsonl") or getattr(f, "path", "").endswith(".json"))
            and (getattr(f, "size", 0) or 0) > 0
        ],
        key=lambda x: x[0],
    )
    pending = [(p, sz) for p, sz in remote_files if p not in done_set]
    if args.max_files > 0:
        pending = pending[:args.max_files]

    print(f"[HF] Tong file: {len(remote_files)} | Da xu ly: {len(done_set)} | Con lai can nen: {len(pending)}", flush=True)
    if not pending:
        print("[Done] Khong co file moi nao can xu ly.")
        return

    staged_shards: List[Path] = []
    staged_sources: List[str] = []
    staged_raw_pos = 0
    staged_uniq_pos = 0
    part_num = int(manifest.get("next_part", 1))

    max_batch_bytes = int(max(1.0, args.max_batch_gb) * 1024 * 1024 * 1024)
    p_idx = 0
    batch_num = 0
    while p_idx < len(pending):
        if args.time_budget > 0 and (time.time() - t_start) >= args.time_budget:
            print("[Time Budget] Sap het thoi gian phien -> dung tai file moi va day not phan da nen.", flush=True)
            break

        chunk = []
        chunk_bytes = 0
        while p_idx < len(pending) and len(chunk) < args.files_per_shard:
            rel_p, sz = pending[p_idx]
            if chunk and (chunk_bytes + sz) > max_batch_bytes:
                break
            chunk.append((rel_p, sz))
            chunk_bytes += sz
            p_idx += 1

        batch_num += 1
        # Don file cu con sot (neu co) truoc khi tai luot moi
        for stale in raw_dir.glob("*"):
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                pass

        dl_pairs = []
        local_paths = []
        chunk_mb = chunk_bytes / (1024 * 1024)
        for rel_p, _ in chunk:
            lp = raw_dir / Path(rel_p).name
            dl_pairs.append((rel_p, str(lp)))
            local_paths.append(str(lp))

        t0 = time.time()
        print(f"\n[Batch {batch_num}] Dang tai {len(chunk)} file ({chunk_mb:.1f} MB)...", flush=True)
        hf_api.download_bucket_files(args.hf_bucket, dl_pairs)
        print(f"[Batch] Tai xong trong {time.time() - t0:.1f}s. Dang chuyen doi sang NPZ nen...", flush=True)

        t1 = time.time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        shard_name = f"hkv3_p{part_num:03d}_s{len(staged_shards) + 1:02d}_{ts}.npz"
        shard_path = shard_dir / shard_name
        raw_n, uniq_n = build_npz_shard(local_paths, shard_path, dedup=not args.no_dedup, work_dir=work_dir)

        # Xoa ngay file raw jsonl de giai phong o dia Kaggle
        for lp in local_paths:
            try:
                Path(lp).unlink(missing_ok=True)
            except Exception:
                pass

        if uniq_n > 0 and shard_path.exists():
            npz_mb = shard_path.stat().st_size / (1024 * 1024)
            ratio = chunk_mb / max(0.01, npz_mb)
            print(f"[Shard] {shard_name}: {raw_n:,} raw -> {uniq_n:,} unique pos | {npz_mb:.1f} MB (nen {ratio:.2f}x) trong {time.time() - t1:.1f}s", flush=True)
            staged_shards.append(shard_path)
            staged_sources.extend(rel_p for rel_p, _ in chunk)
            staged_raw_pos += raw_n
            staged_uniq_pos += uniq_n
        else:
            print("[Shard Warning] Khong trich xuat duoc ban ghi hop le nao trong cum nay.", flush=True)

        # Neu da gom du so shard -> dang ngay len Kaggle Dataset
        if len(staged_shards) >= args.shards_per_dataset:
            if not args.dry_run:
                ds_ref = publish_shards_to_kaggle(k_api, owner, args.dataset_prefix, part_num, staged_shards, args.public)
                manifest["processed_files"].extend(staged_sources)
                manifest["parts"].append({"part": part_num, "dataset": ds_ref, "files": len(staged_sources), "shards": len(staged_shards), "raw_positions": staged_raw_pos, "unique_positions": staged_uniq_pos})
                part_num += 1
                manifest["next_part"] = part_num
                save_and_push_manifest(k_api, index_ref, manifest, work_dir)
            else:
                print(f"[Dry-Run] Bo qua upload {len(staged_shards)} shards len Kaggle.", flush=True)
                for sp in staged_shards:
                    sp.unlink(missing_ok=True)
            staged_shards.clear()
            staged_sources.clear()
            staged_raw_pos = 0
            staged_uniq_pos = 0

    # Day not phan shards con lai neu co
    if staged_shards:
        if not args.dry_run:
            ds_ref = publish_shards_to_kaggle(k_api, owner, args.dataset_prefix, part_num, staged_shards, args.public)
            manifest["processed_files"].extend(staged_sources)
            manifest["parts"].append({"part": part_num, "dataset": ds_ref, "files": len(staged_sources), "shards": len(staged_shards), "raw_positions": staged_raw_pos, "unique_positions": staged_uniq_pos})
            part_num += 1
            manifest["next_part"] = part_num
            save_and_push_manifest(k_api, index_ref, manifest, work_dir)
        else:
            print(f"[Dry-Run] Con du {len(staged_shards)} shards cuoi phien.", flush=True)
            for sp in staged_shards:
                sp.unlink(missing_ok=True)

    print(f"\n[Completed] Tong thoi gian chay: {time.time() - t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
