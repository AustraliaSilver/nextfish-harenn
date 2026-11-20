#!/usr/bin/env python3
"""
================================================================================
HalfKPv3 (HKV3) + HARENN Unified Training Pipeline (Kaggle T4x2 Optimized)
================================================================================
Specification Reference: algorithms/HalfKPv3.md

Architecture Summary (Single Pure Big Net ~88.85 MB):
  1. Input Space (HalfKAv2_hm 11-Channel Canonical Horizontal Mirror):
     - 32 Mirrored King Buckets (files a..d) x 11 Channels x 64 Squares = 22,528 features.
     - O(1) incremental updates (2..3 DirtyPieces per move in C++).
  2. Tri-Scale Virtual Factorizer (Zero Runtime Overhead):
     - Trains 3 resolution spaces simultaneously via direct EmbeddingBag sum:
         W_eff(k, c, s) = W_full[k, c, s] + W_piece[c, s] + W_rel[c, dx+3, dy+7]
       where dx in [-3, +7] (11 cols) and dy in [-7, +7] (15 rows) are computed
       AFTER canonical horizontal King-mirroring.
     - Folded 100% into a single flat int16 table (22,528 x 2,048 = 88.00 MB) on export.
  3. Pairwise Bilinear Gating (2048 -> 1024 per perspective):
     - u[i] = clamp(acc_us[i], 0, 1) * clamp(acc_us[i + 1024], 0, 1) in [0, 1]^1024
     - Native 2nd-order conjunctions (AND gates for pins, x-rays, open files).
  4. Safe Unsigned Parity Streams & Siamese Perspective Twin FC2:
     - u, v in [0, 1]^1024 (maps directly to uint8_t u_u8, v_u8 in [0, 127] for _mm256_maddubs_epi16,
       preserving full 7-bit resolution with max pair sum 2*127*127 = 32,258 <= 32,767).
     - z_us   = FC2_self(u) + FC2_opp(v) + b_fc2
     - z_them = FC2_self(v) + FC2_opp(u) + b_fc2
     - h_us   = SCReLU(z_us), h_them = SCReLU(z_them)
     - h_odd  = h_us - h_them  --> eval_head (NO BIAS) -> Bit-Exact E(Flip(s)) == -E(s)!
     - h_even = h_us + h_them  --> tac_head  (WITH BIAS) -> Bit-Exact [tau,rho,rs](Flip(s)) == +[tau,rho,rs](s)!
  5. 4-Channel Multi-Task PSQT Skip Connection (22,528 x 4):
     - Channel 0 (Odd):    psqt[us][0] - psqt[them][0] added to eval.
     - Channels 1..3 (Even): psqt[us][c] + psqt[them][c] added to tau, rho, rs logits.

Kaggle T4x2 Quickstart:
  python "Neuron Network/train_halfkpv3.py" \
      --sync-hf \
      --hf-bucket australia1/myrandombucket \
      --data data \
      --epochs 16 \
      --finetune-eval-epochs 4 \
      --batch-size 16384 \
      --output nextfish_hkv3.harenn
================================================================================
"""

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════════
# 1. HẰNG SỐ KIẾN TRÚC & LƯỢNG TỬ HÓA PHẦN CỨNG AVX2 (HKV3)
# ═══════════════════════════════════════════════════════════════════════════
HKV3_MAGIC = b"HKV3"
HKV3_ARCH_ID = 3
HKV3_KING_BUCKETS = 32
HKV3_CHANNELS = 11                      # 0..4: Friendly P..Q, 5..9: Enemy P..Q, 10: Enemy K
HKV3_PLANES_PER_BUCKET = HKV3_CHANNELS * 64  # 704
HKV3_NUM_FEATURES = HKV3_KING_BUCKETS * HKV3_PLANES_PER_BUCKET  # 22,528
HKV3_NUM_PIECE_FEATURES = HKV3_CHANNELS * 64                    # 704
HKV3_NUM_REL_FEATURES = HKV3_CHANNELS * 11 * 15                 # 1,815

HKV3_L1_SIZE = 2048                     # Accumulator width per perspective
HKV3_L1_PAIR_SIZE = 1024                # After Pairwise Bilinear Gating (2048 -> 1024)
HKV3_L2_SIZE = 256                      # Siamese Perspective Twin FC2 width
HKV3_PSQT_BUCKETS = 4                   # [eval_psqt, tau_psqt, rho_psqt, rs_psqt]

MAX_ACTIVE_FEATURES = 32                # Max non-own-king pieces on board (<= 31 in standard chess)
PADDING_FEATURE_IDX = HKV3_NUM_FEATURES          # 22,528
PADDING_PIECE_IDX = HKV3_NUM_PIECE_FEATURES      # 704
PADDING_REL_IDX = HKV3_NUM_REL_FEATURES          # 1,815

# Hằng số lượng tử hóa khớp tuyệt đối với C++ AVX2 (_mm256_maddubs_epi16)
QA_SCALE = 127.0                        # L1 accumulator & gated activation scale [0, 127]
QW_FC2_SCALE = 64.0                     # FC2 int8 weight scale [-127, +127] -> max float +-1.984375
FC2_SHIFT = 6                           # 2^6 = 64
Q_HEAD_SCALE = 64.0                     # L3 int16 readout head weight scale
OUTPUT_SCALE = QA_SCALE * QA_SCALE * Q_HEAD_SCALE  # 127 * 127 * 64 = 1,032,256.0
WDL_SCALE_PAWNS = 2.60                  # 2.60 pawns = 260 cp (khớp chuẩn WDL-normalized cp của Stockfish 19)

# Bảng ánh xạ ký tự FEN -> kênh quân cờ cho góc nhìn Trắng (0) và Đen (1)
P2CH_WHITE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4,
    'p': 5, 'n': 6, 'b': 7, 'r': 8, 'q': 9, 'k': 10
}
P2CH_BLACK = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4,
    'P': 5, 'N': 6, 'B': 7, 'R': 8, 'Q': 9, 'K': 10
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. BỘ ÁNH XẠ ĐẶC TRƯNG CHUẨN HÓA GẬP GƯƠNG & VIRTUAL FACTORIZER MAPS
# ═══════════════════════════════════════════════════════════════════════════
def make_hkv3_feature_index(perspective: int, ksq: int, channel: int, sq: int) -> int:
    """
    Tính chỉ số đặc trưng HalfKPv3 chuẩn hóa gập gương ngang (0..22527).
    Khớp bit-for-bit với hàm C++ make_hkv3_feature_index() trong algorithms/HalfKPv3.md:
      1. Định hướng dọc theo màu quân (perspective == 1 -> ^ 56)
      2. Gập gương ngang nếu Vua nằm ở cột e..h ((k_orient & 7) >= 4 -> ^ 7)
      3. King bucket = (k_norm >> 3) * 4 + (k_norm & 3) in [0, 31]
      4. Index = king_bucket * 704 + channel * 64 + s_norm
    """
    k_orient = ksq if perspective == 0 else (ksq ^ 56)
    s_orient = sq if perspective == 0 else (sq ^ 56)

    mirror = 7 if (k_orient & 7) >= 4 else 0
    k_norm = k_orient ^ mirror
    s_norm = s_orient ^ mirror

    king_bucket = (k_norm >> 3) * 4 + (k_norm & 3)
    return king_bucket * HKV3_PLANES_PER_BUCKET + channel * 64 + s_norm


def fen_to_hkv3_sparse_pair(fen: str, stm_override: Optional[object] = None) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Chuyển chuỗi FEN thành cặp mảng chỉ số thưa uint16[32] (us_indices, them_indices, stm).
    """
    parts = fen.split()
    if not parts:
        return None
    board_part = parts[0]
    rows = board_part.split('/')
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

    # Góc nhìn Trắng (perspective = 0)
    w_mirror = 7 if (w_ksq & 7) >= 4 else 0
    w_knorm = w_ksq ^ w_mirror
    w_kb = ((w_knorm >> 3) * 4 + (w_knorm & 3)) * HKV3_PLANES_PER_BUCKET

    # Góc nhìn Đen (perspective = 1)
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
    else:
        return b_arr, w_arr, 1


def build_hkv3_canonical_factorizer_maps() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tiền tính toán bảng tra chỉ số cho Tri-Scale Virtual Factorizer (kèm phần tử padding ở cuối):
      - piece_map_pad: [22,529] -> ánh xạ f in [0..22527] sang c*64 + s_norm in [0..703], và 22528 -> 704.
      - rel_map_pad:   [22,529] -> ánh xạ f in [0..22527] sang c*165 + (dx+3)*15 + (dy+7) in [0..1814], và 22528 -> 1815.
    Lưu ý quan trọng (Mục 4.1 tài liệu HalfKPv3.md):
      dx = file(s_norm) - file(k_norm) in [-3, +7] được tính SAU KHI ĐÃ GẬP GƯƠNG cả k và s,
      đảm bảo dx > 0 luôn hướng về trung tâm / cánh đối diện nhất quán tuyệt đối.
    """
    piece_map = np.full(HKV3_NUM_FEATURES + 1, PADDING_PIECE_IDX, dtype=np.int64)
    rel_map = np.full(HKV3_NUM_FEATURES + 1, PADDING_REL_IDX, dtype=np.int64)

    for kb in range(HKV3_KING_BUCKETS):
        k_rank = kb // 4
        k_file = kb % 4  # Đã gập gương nên k_file luôn thuộc {0, 1, 2, 3}
        for c in range(HKV3_CHANNELS):
            for sq in range(64):
                s_rank = sq // 8
                s_file = sq % 8
                feat_idx = kb * HKV3_PLANES_PER_BUCKET + c * 64 + sq

                piece_map[feat_idx] = c * 64 + sq

                dx_idx = (s_file - k_file) + 3  # [-3..+7] + 3 -> [0..10] (11 cột)
                dy_idx = (s_rank - k_rank) + 7  # [-7..+7] + 7 -> [0..14] (15 hàng)
                rel_map[feat_idx] = c * (11 * 15) + dx_idx * 15 + dy_idx

    return torch.from_numpy(piece_map), torch.from_numpy(rel_map)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ĐỒNG BỘ HUGGING FACE BUCKET & BỘ XÂY DỰNG CACHE ĐA NHÂN (.NPZ)
# ═══════════════════════════════════════════════════════════════════════════
def sync_from_huggingface(bucket_id: str, data_dir: str, token: Optional[str] = None) -> int:
    """Tải toàn bộ các file .jsonl mới từ Hugging Face Bucket về thư mục data_dir."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError("Vui lòng cài đặt huggingface_hub: pip install -U huggingface_hub") from e

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        # Thử đọc từ Kaggle Secrets nếu đang chạy trên Kaggle Notebook
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            print("[HF-Sync] Loaded HF_TOKEN from Kaggle Secrets.")
        except Exception:
            pass

    api = HfApi(token=hf_token)
    print(f"[HF-Sync] Đang kiểm tra danh sách file trên bucket '{bucket_id}'...", flush=True)
    files = list(api.list_bucket_tree(bucket_id))
    jsonl_files = [f for f in files if getattr(f, "path", "").endswith(".jsonl")]
    total_remote_bytes = sum(getattr(f, "size", 0) or 0 for f in jsonl_files)
    print(f"[HF-Sync] Tìm thấy {len(jsonl_files)} file .jsonl ({total_remote_bytes / (1024 * 1024):.2f} MB)", flush=True)

    to_download = []
    for f in jsonl_files:
        rel_path = getattr(f, "path", "")
        remote_size = getattr(f, "size", 0) or 0
        local_file = data_path / Path(rel_path).name
        if local_file.exists() and local_file.stat().st_size == remote_size:
            continue
        to_download.append((rel_path, str(local_file)))

    if not to_download:
        print("[HF-Sync] Toàn bộ file .jsonl ở local đã đầy đủ và mới nhất!", flush=True)
        return len(jsonl_files)

    print(f"[HF-Sync] Đang tải {len(to_download)} file mới về {data_path}...", flush=True)
    t0 = time.time()
    api.download_bucket_files(bucket_id, to_download)
    print(f"[HF-Sync] Hoàn tất tải {len(to_download)} file trong {time.time() - t0:.2f}s!", flush=True)
    return len(jsonl_files)


def _parse_single_data_file_worker(task):
    """Hàm worker chạy đa tiến trình để phân tích 1 file .jsonl hoặc .json sang chỉ số HKV3."""
    if isinstance(task, tuple):
        file_path_str, is_val_file, tail_val_split = task
    else:
        file_path_str, is_val_file, tail_val_split = task, False, 0.0
    path = Path(file_path_str)
    us_list = []
    them_list = []
    eval_list = []
    wdl_list = []
    eval_mask_list = []
    tau_list = []
    rho_list = []
    rs_list = []
    hash_list = []

    total_lines = 0
    parsed_lines = 0
    sample_first_dict = None

    def process_record(pos: dict):
        nonlocal sample_first_dict, parsed_lines
        if not isinstance(pos, dict):
            return

        if sample_first_dict is None:
            sample_first_dict = pos

        # Hỗ trợ linh hoạt các biến thể tên trường FEN
        fen = pos.get("fen") or pos.get("FEN") or pos.get("fen_string") or pos.get("board")
        if not fen or not isinstance(fen, str):
            return

        parts = fen.split()
        if len(parts) < 2:
            return

        # Bao gồm quân (0), lượt đi (1) và quyền nhập thành (2) để phân biệt thế cờ
        board_key = " ".join(parts[:4]) if len(parts) >= 4 else fen
        # Dùng hàm băm mật mã 64-bit blake2b triệt tiêu 100% rủi ro đụng độ birthday collision trên 50M-100M vị trí
        h = struct.unpack("<q", hashlib.blake2b(board_key.encode("utf-8"), digest_size=8).digest())[0]

        # Không dùng toán tử `or` vì pos["stm"] == 0 (Trắng đi) sẽ bị coi là False!
        stm = None
        for k in ("stm", "turn", "side_to_move", "to_move"):
            if k in pos and pos[k] is not None:
                stm = pos[k]
                break
        parsed = fen_to_hkv3_sparse_pair(fen, stm)
        if parsed is None:
            return
        us_arr, them_arr, _ = parsed

        # Hỗ trợ linh hoạt các biến thể tên trường điểm Eval
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

        # Lọc mặt nạ Eval: chỉ train Static Eval trên thế cờ yên tĩnh (is_quiet) và không phải thế Mate (|eval| < 9000).
        # Các thế non-quiet vẫn giữ nguyên để train 3 đầu chiến thuật tau / rho / rs!
        q_raw = pos.get("is_quiet", True)
        if isinstance(q_raw, str):
            is_quiet_flag = q_raw.strip().lower() in ("true", "1", "yes")
        else:
            is_quiet_flag = bool(q_raw)
        eval_mask = 1.0 if (is_quiet_flag and abs(ev_cp_raw) < 9000.0) else 0.0

        ev_cp = max(-1000.0, min(1000.0, ev_cp_raw))
        ev_pawns = ev_cp / 100.0

        # Kết quả ván cờ thật (WDL theo góc nhìn bên đi): 1.0=Win, 0.5=Draw, 0.0=Loss
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

        tau = max(0.0, min(1.0, float(pos.get("tau", 0.5))))
        rho = max(0.0, min(1.0, float(pos.get("rho", 0.3))))
        rs = max(0.0, min(1.0, float(pos.get("rs", 0.5))))

        hash_list.append(h)
        us_list.append(us_arr)
        them_list.append(them_arr)
        eval_list.append(ev_pawns)
        wdl_list.append(wdl_val)
        eval_mask_list.append(eval_mask)
        tau_list.append(tau)
        rho_list.append(rho)
        rs_list.append(rs)
        parsed_lines += 1

    try:
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            positions = data.get("positions", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            total_lines = len(positions)
            for pos in positions:
                process_record(pos)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_lines += 1
                    try:
                        pos = json.loads(line)
                        process_record(pos)
                    except Exception:
                        continue
    except Exception as e:
        print(f"[Worker Error] Lỗi đọc file {path.name}: {e}", flush=True)

    if total_lines > 0 and parsed_lines == 0:
        found_keys = list(sample_first_dict.keys()) if sample_first_dict else []
        print(f"[Schema Warning] File {path.name}: Đọc {total_lines} dòng nhưng 0 dòng parse được! "
              f"Keys tìm thấy: {found_keys}. Yêu cầu tối thiểu: 'fen' và 'eval_score' (hoặc 'eval'/'score'/'cp').", flush=True)

    if not us_list:
        return None

    n_rec = len(eval_list)
    if is_val_file:
        val_arr = np.ones(n_rec, dtype=np.uint8)
    elif tail_val_split > 0.0:
        val_arr = np.zeros(n_rec, dtype=np.uint8)
        n_tail = max(1, int(n_rec * tail_val_split))
        val_arr[n_rec - n_tail:] = 1
    else:
        val_arr = np.zeros(n_rec, dtype=np.uint8)

    return (
        np.asarray(hash_list, dtype=np.int64),
        np.vstack(us_list).astype(np.uint16, copy=False),
        np.vstack(them_list).astype(np.uint16, copy=False),
        np.asarray(eval_list, dtype=np.float32),
        np.asarray(wdl_list, dtype=np.float32),
        np.asarray(eval_mask_list, dtype=np.float32),
        np.asarray(tau_list, dtype=np.float32),
        np.asarray(rho_list, dtype=np.float32),
        np.asarray(rs_list, dtype=np.float32),
        val_arr,
    )


def load_or_build_hkv3_cache(
    data_dir: str,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
    val_split: float = 0.03,
):
    """
    Tải cache nhị phân HKV3 (.npz) hoặc xây dựng song song trên toàn bộ các nhân CPU.
    """
    data_path = Path(data_dir)
    if cache_path:
        cpath = Path(cache_path)
        if not cpath.exists():
            raise FileNotFoundError(
                f"[Cache] Không tìm thấy file cache được chỉ định: {cpath}\n"
                f"Gợi ý: Chạy lệnh `!find /kaggle/input -name '*.npz'` để kiểm tra đường dẫn chính xác của file dataset trên Kaggle."
            )
    elif data_path.is_file() and data_path.suffix == ".npz":
        cpath = data_path
    elif data_path.is_dir():
        default_cache = data_path / "_hkv3_sparse_cache_v2.npz"
        if default_cache.exists():
            cpath = default_cache
        else:
            cpath = default_cache
    else:
        cpath = data_path / "_hkv3_sparse_cache_v2.npz"

    if cpath.exists() and use_cache:
        t0 = time.time()
        print(f"[Cache] Đang nạp cache nhị phân HKV3 từ {cpath}...", flush=True)
        try:
            data = np.load(str(cpath), allow_pickle=False)
            us_indices = data["us_indices"]
            them_indices = data["them_indices"]
            labels_eval = data["labels_eval"]
            if "labels_wdl" in data:
                labels_wdl = data["labels_wdl"]
            else:
                labels_wdl = (1.0 / (1.0 + np.exp(-np.clip(labels_eval / WDL_SCALE_PAWNS, -20.0, 20.0)))).astype(np.float32)
            if "labels_eval_mask" in data:
                labels_eval_mask = data["labels_eval_mask"]
            else:
                labels_eval_mask = np.ones(len(labels_eval), dtype=np.float32)
            labels_tau = data["labels_tau"]
            labels_rho = data["labels_rho"]
            labels_rs = data["labels_rs"]
            is_val_mask = data["is_val"] if "is_val" in data else None

            # Tự động phát hiện và chuẩn hóa nếu file cache được sinh ở không gian 64-King cũ
            max_feat = max(int(us_indices.max()), int(them_indices.max()))
            if max_feat > HKV3_NUM_FEATURES:
                print(f"[Cache Auto-Fix] Phát hiện file cache không gian 64-King (max={max_feat})...", flush=True)
                print(f"[Cache Auto-Fix] Đang tự động chuyển đổi sang 32 Mirrored King Buckets chuẩn HalfKPv3...", flush=True)
                t_fix = time.time()
                lut = np.full(65536, PADDING_FEATURE_IDX, dtype=np.uint16)
                for f in range(64 * HKV3_PLANES_PER_BUCKET):
                    ksq = f // HKV3_PLANES_PER_BUCKET
                    rem = f % HKV3_PLANES_PER_BUCKET
                    ch = rem // 64
                    sq = rem % 64
                    mirror = 7 if (ksq & 7) >= 4 else 0
                    knorm = ksq ^ mirror
                    kb = ((knorm >> 3) * 4 + (knorm & 3)) * HKV3_PLANES_PER_BUCKET
                    lut[f] = kb + ch * 64 + (sq ^ mirror)
                us_indices = lut[us_indices]
                them_indices = lut[them_indices]
                print(f"[Cache Auto-Fix] Chuẩn hóa thành công trong {time.time() - t_fix:.2f}s! (Max index mới: {max(int(us_indices.max()), int(them_indices.max()))})", flush=True)

            quiet_pct = 100.0 * float(np.mean(labels_eval_mask))
            print(f"[Cache] Đã nạp {len(labels_eval):,} thế cờ duy nhất (Quiet Eval Mask: {quiet_pct:.1f}%) trong {time.time() - t0:.2f}s!", flush=True)
            return us_indices, them_indices, labels_eval, labels_wdl, labels_eval_mask, labels_tau, labels_rho, labels_rs, is_val_mask
        except Exception as e:
            print(f"[Cache] File cache cũ hoặc lỗi ({e}), tiến hành xây dựng lại...", flush=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_path}")

    source_files = sorted(list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json")))
    if not source_files:
        raise RuntimeError(f"Không tìm thấy bất kỳ file .jsonl hay .json nào trong {data_path}")

    workers = max(1, min(8, cpu_count()))
    print(f"[Dataset] Đang phân tích {len(source_files)} file dữ liệu sang không gian HalfKPv3 (22,528 features) "
          f"với {workers} tiến trình CPU...", flush=True)
    t0 = time.time()

    # Tách Val độc lập theo file (nếu >= 10 file) hoặc theo phần đuôi ván đấu (nếu < 10 file) để chống data leakage cùng ván!
    if len(source_files) >= 10:
        n_val_files = max(1, int(round(len(source_files) * val_split)))
        tasks = [(str(f), i >= len(source_files) - n_val_files, 0.0) for i, f in enumerate(source_files)]
    else:
        tasks = [(str(f), False, val_split) for f in source_files]

    with Pool(workers) as pool:
        results = pool.map(_parse_single_data_file_worker, tasks)

    valid_results = [r for r in results if r is not None]
    del results
    if not valid_results:
        raise RuntimeError("Không trích xuất được thế cờ hợp lệ nào từ các file dữ liệu!")

    all_hashes = np.concatenate([r[0] for r in valid_results], axis=0)
    all_us = np.concatenate([r[1] for r in valid_results], axis=0)
    all_them = np.concatenate([r[2] for r in valid_results], axis=0)
    all_eval = np.concatenate([r[3] for r in valid_results], axis=0)
    all_wdl = np.concatenate([r[4] for r in valid_results], axis=0)
    all_eval_mask = np.concatenate([r[5] for r in valid_results], axis=0)
    all_tau = np.concatenate([r[6] for r in valid_results], axis=0)
    all_rho = np.concatenate([r[7] for r in valid_results], axis=0)
    all_rs = np.concatenate([r[8] for r in valid_results], axis=0)
    all_is_val = np.concatenate([r[9] for r in valid_results], axis=0)
    del valid_results
    gc.collect()

    total_raw = len(all_eval)
    _, unique_idx, inv_idx, counts = np.unique(
        all_hashes, return_index=True, return_inverse=True, return_counts=True
    )
    del all_hashes

    us_indices = all_us[unique_idx]
    del all_us
    them_indices = all_them[unique_idx]
    del all_them, unique_idx

    # Nếu một thế cờ xuất hiện ở cả tập Train và Val, ép thuộc về Train để 0% thế cờ trùng lọt sang Val
    train_hits = np.bincount(inv_idx, weights=(1 - all_is_val).astype(np.int32))
    is_val_mask = (train_hits == 0).astype(np.uint8)
    del all_is_val, train_hits

    counts_f = counts.astype(np.float32)
    del counts
    labels_eval = (np.bincount(inv_idx, weights=all_eval) / counts_f).astype(np.float32)
    del all_eval
    labels_wdl = (np.bincount(inv_idx, weights=all_wdl) / counts_f).astype(np.float32)
    del all_wdl
    labels_eval_mask = ((np.bincount(inv_idx, weights=all_eval_mask) / counts_f) >= 0.5).astype(np.float32)
    del all_eval_mask
    labels_tau = (np.bincount(inv_idx, weights=all_tau) / counts_f).astype(np.float32)
    del all_tau
    labels_rho = (np.bincount(inv_idx, weights=all_rho) / counts_f).astype(np.float32)
    del all_rho
    labels_rs = (np.bincount(inv_idx, weights=all_rs) / counts_f).astype(np.float32)
    del all_rs, inv_idx, counts_f
    gc.collect()

    dt = time.time() - t0
    quiet_pct = 100.0 * float(np.mean(labels_eval_mask))
    print(f"[Dataset] Hoàn tất lọc trùng: giữ lại {len(labels_eval):,} thế cờ duy nhất "
          f"(từ {total_raw:,} bản ghi thô | Quiet Eval: {quiet_pct:.1f}%) trong {dt:.2f}s ({total_raw / max(1e-6, dt):,.0f} pos/s).", flush=True)

    if use_cache:
        try:
            t_save = time.time()
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(cpath),
                us_indices=us_indices,
                them_indices=them_indices,
                labels_eval=labels_eval,
                labels_wdl=labels_wdl,
                labels_eval_mask=labels_eval_mask,
                labels_tau=labels_tau,
                labels_rho=labels_rho,
                labels_rs=labels_rs,
                is_val=is_val_mask,
            )
            sz_mb = cpath.stat().st_size / (1024 * 1024)
            print(f"[Cache] Đã lưu cache nhị phân tốc độ cao tại {cpath} ({sz_mb:.1f} MB, {time.time() - t_save:.1f}s).", flush=True)
        except Exception as e:
            print(f"[Cache] Cảnh báo không thể ghi file cache: {e}", flush=True)

    return us_indices, them_indices, labels_eval, labels_wdl, labels_eval_mask, labels_tau, labels_rho, labels_rs, is_val_mask


# ═══════════════════════════════════════════════════════════════════════════
# 4. BỘ NẠP BATCH SIÊU TỐC TRÊN GPU VỚI PINNED MEMORY DMA
# ═══════════════════════════════════════════════════════════════════════════
class FastHKV3BatchLoader:
    """
    Bộ nạp dữ liệu trực tiếp từ bộ nhớ NumPy sang GPU với:
      - Pinned Memory DMA (bất đồng bộ thực sự qua PCIe với non_blocking=True).
      - Truyền uint16 (2 bytes) qua PCIe rồi ép kiểu int64 ngay trên GPU VRAM (giảm 75% băng thông PCIe).
      - Mặc định symmetry=1 vì kiến trúc HalfKPv3 đã thỏa mãn đối xứng đại số tuyệt đối (E(them, us) == -E(us, them)).
    """

    def __init__(
        self,
        us_data: np.ndarray,
        them_data: np.ndarray,
        eval_data: np.ndarray,
        wdl_data: np.ndarray,
        eval_mask_data: np.ndarray,
        tau_data: np.ndarray,
        rho_data: np.ndarray,
        rs_data: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        symmetry: int = 1,
        device: torch.device = torch.device("cuda"),
    ):
        self.us_data = us_data
        self.them_data = them_data
        self.eval_data = eval_data
        self.wdl_data = wdl_data
        self.eval_mask_data = eval_mask_data
        self.tau_data = tau_data
        self.rho_data = rho_data
        self.rs_data = rs_data
        self.base_indices = np.asarray(indices, dtype=np.int64)
        self.num_base = len(self.base_indices)
        self.symmetry = symmetry if symmetry in (1, 2) else 1
        self.total_samples = self.num_base * self.symmetry
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.pin_memory = (self.device.type == "cuda")
        self._pin_bufs: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return (self.total_samples + self.batch_size - 1) // self.batch_size

    def release(self):
        """Giải phóng ngay các mảng NumPy & pinned buffer của chunk trước khi nạp chunk mới."""
        self.us_data = None
        self.them_data = None
        self.eval_data = None
        self.wdl_data = None
        self.eval_mask_data = None
        self.tau_data = None
        self.rho_data = None
        self.rs_data = None
        self.base_indices = None
        self._pin_bufs.clear()

    def _to_gpu(self, key: str, arr: np.ndarray, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if self.pin_memory:
            buf = self._pin_bufs.get(key)
            if buf is None or buf.shape != arr.shape:
                buf = torch.empty(arr.shape, dtype=torch.from_numpy(arr[:0]).dtype, pin_memory=True)
                self._pin_bufs[key] = buf
            buf.numpy()[...] = arr
            t_gpu = buf.to(self.device, non_blocking=True)
        else:
            t_gpu = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        if target_dtype is not None and t_gpu.dtype != target_dtype:
            t_gpu = t_gpu.to(target_dtype)
        return t_gpu

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.total_samples)
        else:
            perm = np.arange(self.total_samples, dtype=np.int64)

        for start in range(0, self.total_samples, self.batch_size):
            v_ids = perm[start:start + self.batch_size]
            if self.symmetry == 2:
                is_flipped = (v_ids >= self.num_base)
                raw_pos = v_ids % self.num_base
                actual_ids = self.base_indices[raw_pos]
                us_np = self.us_data[actual_ids]
                them_np = self.them_data[actual_ids]
                ev_np = self.eval_data[actual_ids]
                wdl_np = self.wdl_data[actual_ids]
                if np.any(is_flipped):
                    us_np = np.where(is_flipped[:, None], self.them_data[actual_ids], us_np)
                    them_np = np.where(is_flipped[:, None], self.us_data[actual_ids], them_np)
                    ev_np = np.where(is_flipped, -ev_np, ev_np)
                    wdl_np = np.where(is_flipped, 1.0 - wdl_np, wdl_np)
            else:
                actual_ids = self.base_indices[v_ids]
                us_np = self.us_data[actual_ids]
                them_np = self.them_data[actual_ids]
                ev_np = self.eval_data[actual_ids]
                wdl_np = self.wdl_data[actual_ids]

            # Truyền uint16 (2 bytes) qua PCIe bằng Pinned Memory DMA, sau đó mới .to(torch.int64) trên GPU VRAM:
            us_t = self._to_gpu("us", us_np, target_dtype=torch.int64)
            them_t = self._to_gpu("them", them_np, target_dtype=torch.int64)

            targets = {
                "eval": self._to_gpu("eval", ev_np),
                "wdl": self._to_gpu("wdl", wdl_np),
                "eval_mask": self._to_gpu("eval_mask", self.eval_mask_data[actual_ids]),
                "tau": self._to_gpu("tau", self.tau_data[actual_ids]),
                "rho": self._to_gpu("rho", self.rho_data[actual_ids]),
                "rs": self._to_gpu("rs", self.rs_data[actual_ids]),
            }
            yield us_t, them_t, targets


def _peek_npz_rows(npz_path: Path) -> int:
    """Đọc nhanh số hàng N từ header .npy trong .npz mà không giải nén toàn bộ mảng."""
    with zipfile.ZipFile(str(npz_path), "r") as zf:
        target = "labels_eval.npy" if "labels_eval.npy" in zf.namelist() else zf.namelist()[0]
        with zf.open(target, "r") as fp:
            ver = np.lib.format.read_magic(fp)
            if ver == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(fp)
            elif ver == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(fp)
            else:
                shape, _, _ = np.lib.format._read_array_header(fp, ver)
            return int(shape[0])


def _load_single_npz_arrays(cpath: Path, max_rows: int = 0) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Đọc 1 file .npz HKV3 chuẩn, tự chuẩn hóa 64-king nếu gặp file đời cũ."""
    with np.load(str(cpath), allow_pickle=False) as data:
        n = len(data["labels_eval"])
        sl = slice(0, min(n, max_rows)) if max_rows > 0 else slice(None)
        us_indices = np.asarray(data["us_indices"][sl], dtype=np.uint16)
        them_indices = np.asarray(data["them_indices"][sl], dtype=np.uint16)
        labels_eval = np.asarray(data["labels_eval"][sl], dtype=np.float32)
        if "labels_wdl" in data:
            labels_wdl = np.asarray(data["labels_wdl"][sl], dtype=np.float32)
        else:
            labels_wdl = (1.0 / (1.0 + np.exp(-np.clip(labels_eval / WDL_SCALE_PAWNS, -20.0, 20.0)))).astype(np.float32)
        if "labels_eval_mask" in data:
            labels_eval_mask = np.asarray(data["labels_eval_mask"][sl])
        else:
            labels_eval_mask = np.ones(len(labels_eval), dtype=np.uint8)
        labels_tau = np.asarray(data["labels_tau"][sl], dtype=np.float32)
        labels_rho = np.asarray(data["labels_rho"][sl], dtype=np.float32)
        labels_rs = np.asarray(data["labels_rs"][sl], dtype=np.float32)

    if len(labels_eval) > 0:
        max_feat = max(int(us_indices.max()), int(them_indices.max()))
        if max_feat > HKV3_NUM_FEATURES:
            lut = np.full(65536, PADDING_FEATURE_IDX, dtype=np.uint16)
            for f in range(64 * HKV3_PLANES_PER_BUCKET):
                ksq = f // HKV3_PLANES_PER_BUCKET
                rem = f % HKV3_PLANES_PER_BUCKET
                ch = rem // 64
                sq = rem % 64
                mirror = 7 if (ksq & 7) >= 4 else 0
                knorm = ksq ^ mirror
                kb = ((knorm >> 3) * 4 + (knorm & 3)) * HKV3_PLANES_PER_BUCKET
                lut[f] = kb + ch * 64 + (sq ^ mirror)
            us_indices = lut[us_indices]
            them_indices = lut[them_indices]

    return us_indices, them_indices, labels_eval, labels_wdl, labels_eval_mask, labels_tau, labels_rho, labels_rs


def _load_npz_chunk_prealloc(paths: List[Path]):
    """Nạp danh sách nhiều file .npz bằng cấp phát sẵn 1 lần, tránh tăng gấp đôi RAM như np.concatenate."""
    counts = [_peek_npz_rows(p) for p in paths]
    total = sum(counts)
    us_out = np.empty((total, MAX_ACTIVE_FEATURES), dtype=np.uint16)
    them_out = np.empty((total, MAX_ACTIVE_FEATURES), dtype=np.uint16)
    ev_out = np.empty(total, dtype=np.float32)
    wdl_out = np.empty(total, dtype=np.float32)
    msk_out = np.empty(total, dtype=np.uint8)
    tau_out = np.empty(total, dtype=np.float32)
    rho_out = np.empty(total, dtype=np.float32)
    rs_out = np.empty(total, dtype=np.float32)

    off = 0
    for p, cnt in zip(paths, counts):
        if cnt == 0:
            continue
        u, t, ev, wdl, msk, tau, rho, rs = _load_single_npz_arrays(p)
        end = off + len(ev)
        us_out[off:end] = u
        them_out[off:end] = t
        ev_out[off:end] = ev
        wdl_out[off:end] = wdl
        msk_out[off:end] = msk
        tau_out[off:end] = tau
        rho_out[off:end] = rho
        rs_out[off:end] = rs
        off = end
        del u, t, ev, wdl, msk, tau, rho, rs

    if off < total:
        us_out = us_out[:off]
        them_out = them_out[:off]
        ev_out = ev_out[:off]
        wdl_out = wdl_out[:off]
        msk_out = msk_out[:off]
        tau_out = tau_out[:off]
        rho_out = rho_out[:off]
        rs_out = rs_out[:off]
    return us_out, them_out, ev_out, wdl_out, msk_out, tau_out, rho_out, rs_out


class MultiNPZStreamer:
    """
    Quản lý nạp luân phiên (streaming) cho hàng chục/trăm file .npz (~3+ tỷ thế cờ) vừa khít RAM 30 GB:
      - Chế độ Local Directory (--cache-dir hoặc --data chứa .npz): quét đệ quy *.npz, giữ file cuối làm Val cố định,
        gom các file còn lại thành từng super-chunk <= chunk_max_pos (~80M thế cờ ≈ 12 GB RAM).
      - Chế độ Kaggle Manifest (--kaggle-index): tự đọc packed_manifest.json, giữ part cuối làm Val,
        tải ngầm (prefetch) từng part Public về train rồi xóa ngay khỏi ổ tạm.
    """

    def __init__(
        self,
        local_npz_files: Optional[List[Path]] = None,
        kaggle_index_ref: str = "",
        batch_size: int = 16384,
        symmetry: int = 1,
        chunk_max_pos: int = 80_000_000,
        start_part: int = 0,
        end_part: int = 0,
        device: torch.device = torch.device("cuda"),
        stream_tmp_dir: Optional[Path] = None,
    ):
        self.batch_size = batch_size
        self.symmetry = symmetry
        self.chunk_max_pos = max(1_000_000, int(chunk_max_pos))
        self.device = device
        self.kaggle_index_ref = kaggle_index_ref
        self.stream_tmp_dir = stream_tmp_dir or Path("/tmp/hkv3_stream_npz")
        self.k_api = None
        self.part_entries: List[Dict] = []
        self.local_chunks: List[List[Path]] = []

        if kaggle_index_ref:
            # Thử lấy Kaggle API auth nếu có; nếu không có vẫn tải trực tiếp Public dataset qua HTTPS!
            self.k_api = None
            try:
                if not os.environ.get("KAGGLE_API_TOKEN"):
                    try:
                        from kaggle_secrets import UserSecretsClient
                        t = UserSecretsClient().get_secret("KAGGLE_API_TOKEN") or ""
                        if t:
                            os.environ["KAGGLE_API_TOKEN"] = t
                    except Exception:
                        pass
                from kaggle.api.kaggle_api_extended import KaggleApi
                k = KaggleApi()
                k.authenticate()
                self.k_api = k
            except Exception:
                self.k_api = None

            idx_dir = self.stream_tmp_dir / "_index"
            self._download_public_or_api(kaggle_index_ref, idx_dir)
            manifest = json.loads((idx_dir / "packed_manifest.json").read_text(encoding="utf-8"))
            all_parts = manifest.get("parts", [])
            if start_part > 0:
                all_parts = [p for p in all_parts if int(p["part"]) >= start_part]
            if end_part > 0:
                all_parts = [p for p in all_parts if int(p["part"]) <= end_part]
            if len(all_parts) < 2:
                raise RuntimeError(f"Manifest chỉ có {len(all_parts)} part, cần >= 2 part (hoặc giảm --start-part).")

            val_part = None
            val_files: List[Path] = []
            for cand in reversed(all_parts):
                val_dir = self.stream_tmp_dir / f"val_p{int(cand['part']):03d}"
                print(f"[Stream Val] Đang tải part cố định cho Validation: {cand['dataset']}...", flush=True)
                try:
                    self._download_public_or_api(cand["dataset"], val_dir)
                    vfs = sorted(val_dir.glob("*.npz"))
                    if vfs:
                        val_part = cand
                        val_files = vfs
                        self.val_arrays = _load_single_npz_arrays(val_files[-1], max_rows=100_000)
                        shutil.rmtree(val_dir, ignore_errors=True)
                        break
                except Exception as e:
                    print(f"[Stream Val Warn] Chưa tải được {cand['dataset']} ({e}), thử part trước đó...", flush=True)
                shutil.rmtree(val_dir, ignore_errors=True)

            if val_part is None:
                raise RuntimeError("Không tải được bất kỳ part nào cho Validation từ manifest!")
            self.part_entries = [p for p in all_parts if int(p["part"]) != int(val_part["part"])]
            print(
                f"[Multi-NPZ Kaggle] {len(self.part_entries)} train parts | Val cố định: {len(self.val_arrays[2]):,} thế cờ từ {val_part['dataset']}",
                flush=True,
            )
        else:
            files = sorted(local_npz_files or [])
            if len(files) < 2:
                raise RuntimeError("Cần >= 2 file .npz cho MultiNPZStreamer.")
            val_file = files[-1]
            train_files = files[:-1]
            self.val_arrays = _load_single_npz_arrays(val_file, max_rows=100_000)

            cur_grp: List[Path] = []
            cur_pos = 0
            for f in train_files:
                n = _peek_npz_rows(f)
                if cur_grp and (cur_pos + n > self.chunk_max_pos):
                    self.local_chunks.append(cur_grp)
                    cur_grp = []
                    cur_pos = 0
                cur_grp.append(f)
                cur_pos += n
            if cur_grp:
                self.local_chunks.append(cur_grp)
            total_train = sum(_peek_npz_rows(f) for f in train_files)
            print(
                f"[Multi-NPZ Local] {len(train_files)} file ({total_train:,} pos) -> {len(self.local_chunks)} super-chunks | "
                f"Val cố định: {len(self.val_arrays[2]):,} pos từ {val_file.name}",
                flush=True,
            )

    def get_val_loader(self) -> Tuple[FastHKV3BatchLoader, np.ndarray, np.ndarray]:
        u, t, ev, wdl, msk, tau, rho, rs = self.val_arrays
        idx = np.arange(len(ev), dtype=np.int64)
        loader = FastHKV3BatchLoader(u, t, ev, wdl, msk, tau, rho, rs, idx, batch_size=self.batch_size, shuffle=False, symmetry=1, device=self.device)
        return loader, u, t

    def num_chunks(self) -> int:
        return len(self.part_entries) if self.kaggle_index_ref else len(self.local_chunks)

    def _download_public_or_api(self, ds_ref: str, out_dir: Path) -> Path:
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.k_api is not None:
            try:
                self.k_api.dataset_download_files(ds_ref, path=str(out_dir), unzip=True, quiet=True)
                return out_dir
            except Exception as e:
                print(f"[Stream Warn] Kaggle API download loi ({e}), fallback HTTPS public...", flush=True)

        import urllib.request
        url = f"https://www.kaggle.com/api/v1/datasets/download/{ds_ref}"
        zip_path = out_dir / "_dl.zip"
        for attempt in range(1, 5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=300) as resp, open(zip_path, "wb") as f:
                    while True:
                        buf = resp.read(4 * 1024 * 1024)
                        if not buf:
                            break
                        f.write(buf)
                with zipfile.ZipFile(str(zip_path), "r") as zf:
                    zf.extractall(str(out_dir))
                zip_path.unlink(missing_ok=True)
                return out_dir
            except Exception as e:
                zip_path.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                print(f"[Stream Retry {attempt}/4] Loi tai {ds_ref}: {e}", flush=True)
                time.sleep(5 * attempt)
        return out_dir

    def _download_kaggle_part(self, part_entry: Dict) -> Path:
        pnum = int(part_entry["part"])
        pdir = self.stream_tmp_dir / f"train_p{pnum:03d}"
        return self._download_public_or_api(part_entry["dataset"], pdir)

    def iter_chunks(self, epoch_seed: int, start_chunk: int = 0):
        rng = random.Random(epoch_seed)
        if self.kaggle_index_ref:
            order = list(self.part_entries)
            rng.shuffle(order)
            n_chunks = len(order)
            if start_chunk >= n_chunks:
                return
            with ThreadPoolExecutor(max_workers=1) as dl_pool:
                cur_fut = dl_pool.submit(self._download_kaggle_part, order[start_chunk])
                consec_skips = 0
                for c_idx in range(start_chunk, n_chunks):
                    try:
                        pdir = cur_fut.result()
                    except Exception as e:
                        consec_skips += 1
                        print(f"  [Stream Skip Part {order[c_idx]['part']:03d}] Bỏ qua do lỗi tải ({e})", flush=True)
                        if consec_skips >= 5:
                            raise RuntimeError(f"Đã bỏ qua {consec_skips} part liên tiếp khi tải dữ liệu!") from e
                        pdir = None
                    next_fut = dl_pool.submit(self._download_kaggle_part, order[c_idx + 1]) if (c_idx + 1) < n_chunks else None
                    if pdir is None:
                        cur_fut = next_fut
                        continue
                    consec_skips = 0
                    npz_list = sorted(pdir.glob("*.npz"))
                    if not npz_list:
                        shutil.rmtree(pdir, ignore_errors=True)
                        cur_fut = next_fut
                        continue
                    t_ld = time.time()
                    arrs = _load_npz_chunk_prealloc(npz_list)
                    shutil.rmtree(pdir, ignore_errors=True)
                    u, t, ev, wdl, msk, tau, rho, rs = arrs
                    del arrs
                    print(
                        f"  [Stream Part {order[c_idx]['part']:03d} | Chunk {c_idx + 1}/{n_chunks}] "
                        f"Nạp {len(ev):,} pos trong {time.time() - t_ld:.1f}s",
                        flush=True,
                    )
                    idx = np.arange(len(ev), dtype=np.int64)
                    loader = FastHKV3BatchLoader(u, t, ev, wdl, msk, tau, rho, rs, idx, batch_size=self.batch_size, shuffle=True, symmetry=self.symmetry, device=self.device)
                    yield c_idx, n_chunks, loader
                    del loader, idx, u, t, ev, wdl, msk, tau, rho, rs
                    gc.collect()
                    cur_fut = next_fut
        else:
            order = list(self.local_chunks)
            rng.shuffle(order)
            n_chunks = len(order)
            for c_idx in range(start_chunk, n_chunks):
                t_ld = time.time()
                u, t, ev, wdl, msk, tau, rho, rs = _load_npz_chunk_prealloc(order[c_idx])
                print(
                    f"  [Local Chunk {c_idx + 1}/{n_chunks}] "
                    f"Nạp {len(ev):,} pos ({len(order[c_idx])} shards) trong {time.time() - t_ld:.1f}s",
                    flush=True,
                )
                idx = np.arange(len(ev), dtype=np.int64)
                loader = FastHKV3BatchLoader(u, t, ev, wdl, msk, tau, rho, rs, idx, batch_size=self.batch_size, shuffle=True, symmetry=self.symmetry, device=self.device)
                yield c_idx, n_chunks, loader
                del loader, idx, u, t, ev, wdl, msk, tau, rho, rs
                gc.collect()


class HFTelemetry:
    """Đẩy live status.json, train.log và backup .harenn lên Hugging Face (nếu có HF_TOKEN)."""

    def __init__(self, repo_id: str, live_log_file: str = ""):
        self.repo_id = repo_id.strip()
        self.live_log_file = Path(live_log_file) if live_log_file else None
        self.api = None
        self._pool = None
        self._fut = None
        self._stage_dir = Path("/tmp/hf_telemetry_stage")
        if not self.repo_id:
            return
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient
                token = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
            except Exception:
                token = ""
        if not token:
            print("[HF Live Log] Không tìm thấy HF_TOKEN -> chỉ ghi log local.", flush=True)
            return
        try:
            from huggingface_hub import HfApi
            self.api = HfApi(token=token)
            self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=True, exist_ok=True)
            self._pool = ThreadPoolExecutor(max_workers=1)
            print(f"[HF Live Log] Đã bật đẩy log trực tiếp lên HF dataset: {self.repo_id}", flush=True)
        except Exception as e:
            print(f"[HF Live Log Warn] Không khởi tạo được HF repo ({e}) -> chỉ ghi local.", flush=True)
            self.api = None

    def push(self, status: Dict, extra_files: Optional[List[Path]] = None, wait: bool = False):
        if self.api is None or self._pool is None:
            return
        if self._fut is not None and not self._fut.done():
            if not wait:
                return
            try:
                self._fut.result()
            except Exception:
                pass

        shutil.rmtree(self._stage_dir, ignore_errors=True)
        self._stage_dir.mkdir(parents=True, exist_ok=True)
        st_path = self._stage_dir / "status.json"
        st_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        to_upload: List[Tuple[str, str]] = [(str(st_path), "status.json")]

        if self.live_log_file and self.live_log_file.exists():
            log_cp = self._stage_dir / "train_live.log"
            try:
                shutil.copy2(self.live_log_file, log_cp)
                to_upload.append((str(log_cp), "train_live.log"))
            except Exception:
                pass
        for ef in (extra_files or []):
            if ef and Path(ef).exists():
                dst = self._stage_dir / Path(ef).name
                try:
                    shutil.copy2(ef, dst)
                    to_upload.append((str(dst), Path(ef).name))
                except Exception:
                    pass

        def _do_commit():
            try:
                from huggingface_hub import CommitOperationAdd
                ops = [CommitOperationAdd(path_in_repo=rname, path_or_fileobj=lpath) for lpath, rname in to_upload]
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Update live train log ({status.get('stage', 'init')})",
                )
            except Exception as e:
                print(f"[HF Live Log Warn] Lỗi đẩy HF: {e}", flush=True)

        self._fut = self._pool.submit(_do_commit)
        if wait:
            try:
                self._fut.result()
            except Exception:
                pass

    def close(self):
        if self._fut is not None:
            try:
                self._fut.result()
            except Exception:
                pass
        if self._pool is not None:
            self._pool.shutdown(wait=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. KIẾN TRÚC MẠNG NƠ-RON ĐA NHIỆM HALFKPV3 (HKV3) CHUẨN ĐẠI SỐ & SIMD
# ═══════════════════════════════════════════════════════════════════════════
def fake_quant_ste(x: torch.Tensor, scale: float, min_q: float, max_q: float) -> torch.Tensor:
    """Straight-Through Estimator (STE) mô phỏng lượng tử hóa làm tròn (round) của trọng số C++."""
    q = torch.clamp(torch.round(x * scale), min=min_q, max=max_q) / scale
    return x + (q - x).detach()


def bilinear_gate_ste(x_left: torch.Tensor, x_right: torch.Tensor, qat: bool) -> torch.Tensor:
    """
    Mô phỏng chính xác 100% lệnh Pairwise Bilinear Gating trong C++ AVX2:
      u_u8 = (u_i16 * v_i16) >> 7   (chia cho 128 làm tròn xuống floor, giá trị thuộc [0, 126]).
    """
    prod_cont = (x_left * x_right) * (QA_SCALE / 128.0)
    if not qat:
        return prod_cont
    xl_i16 = torch.clamp(torch.round(x_left.float() * QA_SCALE), 0.0, 127.0)
    xr_i16 = torch.clamp(torch.round(x_right.float() * QA_SCALE), 0.0, 127.0)
    u_u8 = torch.clamp(torch.div(xl_i16 * xr_i16, 128.0, rounding_mode="floor"), 0.0, 127.0)
    u_q = (u_u8 / QA_SCALE).to(prod_cont.dtype)
    return prod_cont + (u_q - prod_cont).detach()


def fc2_screlu_ste(z: torch.Tensor, qat: bool) -> torch.Tensor:
    """
    Mô phỏng chính xác 100% phép dịch bit số nguyên (sum_i32 >> 6) rồi clamp [0, 127] và bình phương SCReLU của C++.
    Thực hiện ở FP32 để tránh tràn số FP16 khi nhân (QA_SCALE * QW_FC2_SCALE = 8128.0).
    """
    z_f32 = z.float()
    z_clamped = torch.clamp(z_f32, 0.0, 1.0)
    if qat:
        z_i32 = torch.round(z_f32 * (QA_SCALE * QW_FC2_SCALE))
        z_shifted = torch.div(z_i32, QW_FC2_SCALE, rounding_mode="floor")
        z_q = torch.clamp(z_shifted, 0.0, 127.0) / QA_SCALE
        z_act = z_clamped + (z_q - z_clamped).detach()
    else:
        z_act = z_clamped
    return z_act.square().to(z.dtype)


_HKV3_CUDA_EXT = None
_HKV3_CUDA_EXT_TRIED = False


def _get_hkv3_cuda_ext():
    """Biên dịch Fused CUDA Sparse L1 Kernel (nếu có nvcc) để tăng tốc 3-6x thay cụm EmbeddingBag."""
    global _HKV3_CUDA_EXT, _HKV3_CUDA_EXT_TRIED
    if _HKV3_CUDA_EXT_TRIED:
        return _HKV3_CUDA_EXT
    _HKV3_CUDA_EXT_TRIED = True
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load_inline
        if not CUDA_HOME and shutil.which("nvcc") is None:
            return None
        cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void hkv3_l1_fwd_kernel(
    const int64_t* __restrict__ idx,
    const int64_t* __restrict__ idx_full,
    const int64_t* __restrict__ piece_map,
    const int64_t* __restrict__ rel_map,
    const float* __restrict__ W_full,
    const float* __restrict__ W_piece,
    const float* __restrict__ W_rel,
    const float* __restrict__ bias,
    float* __restrict__ out,
    uint8_t* __restrict__ clamp_mask,
    int64_t B, int K, int H, int64_t pad_f, bool qat, float qa_scale
) {
    __shared__ int32_t s_f[64];
    __shared__ int32_t s_p[64];
    __shared__ int32_t s_r[64];

    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (tid < K) {
        int64_t f_orig = idx[(int64_t)b * K + tid];
        int64_t f_full = idx_full[(int64_t)b * K + tid];
        s_f[tid] = (f_full >= 0 && f_full < pad_f) ? (int32_t)f_full : -1;
        if (f_orig >= 0 && f_orig < pad_f) {
            s_p[tid] = (int32_t)piece_map[f_orig];
            s_r[tid] = (int32_t)rel_map[f_orig];
        } else {
            s_p[tid] = -1;
            s_r[tid] = -1;
        }
    }
    __syncthreads();

    int d = blockIdx.y * blockDim.x + tid;
    if (d >= H) return;

    float acc = bias[d];
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        int32_t f = s_f[k];
        int32_t p = s_p[k];
        int32_t r = s_r[k];
        if (f >= 0) acc += W_full[(int64_t)f * H + d];
        if (p >= 0) acc += W_piece[(int64_t)p * H + d];
        if (r >= 0) acc += W_rel[(int64_t)r * H + d];
    }
    uint8_t m = (acc > 0.0f && acc < 1.0f) ? 1 : 0;
    float c = acc < 0.0f ? 0.0f : (acc > 1.0f ? 1.0f : acc);
    if (qat) {
        c = rintf(c * qa_scale) / qa_scale;
    }
    out[(int64_t)b * H + d] = c;
    clamp_mask[(int64_t)b * H + d] = m;
}

__global__ void hkv3_l1_bwd_kernel(
    const float* __restrict__ grad_out,
    const uint8_t* __restrict__ clamp_mask,
    const int64_t* __restrict__ idx,
    const int64_t* __restrict__ idx_full,
    const int64_t* __restrict__ piece_map,
    const int64_t* __restrict__ rel_map,
    float* __restrict__ grad_W_full,
    float* __restrict__ grad_W_piece,
    float* __restrict__ grad_W_rel,
    int64_t B, int K, int H, int64_t pad_f
) {
    __shared__ int32_t s_f[64];
    __shared__ int32_t s_p[64];
    __shared__ int32_t s_r[64];

    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (tid < K) {
        int64_t f_orig = idx[(int64_t)b * K + tid];
        int64_t f_full = idx_full[(int64_t)b * K + tid];
        s_f[tid] = (f_full >= 0 && f_full < pad_f) ? (int32_t)f_full : -1;
        if (f_orig >= 0 && f_orig < pad_f) {
            s_p[tid] = (int32_t)piece_map[f_orig];
            s_r[tid] = (int32_t)rel_map[f_orig];
        } else {
            s_p[tid] = -1;
            s_r[tid] = -1;
        }
    }
    __syncthreads();

    int d = blockIdx.y * blockDim.x + tid;
    if (d >= H) return;
    if (!clamp_mask[(int64_t)b * H + d]) return;
    float g = grad_out[(int64_t)b * H + d];
    if (g == 0.0f) return;

    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        int32_t f = s_f[k];
        int32_t p = s_p[k];
        int32_t r = s_r[k];
        if (f >= 0) atomicAdd(&grad_W_full[(int64_t)f * H + d], g);
        if (p >= 0) atomicAdd(&grad_W_piece[(int64_t)p * H + d], g);
        if (r >= 0) atomicAdd(&grad_W_rel[(int64_t)r * H + d], g);
    }
}

std::vector<torch::Tensor> hkv3_l1_forward(
    torch::Tensor idx, torch::Tensor idx_full,
    torch::Tensor piece_map, torch::Tensor rel_map,
    torch::Tensor W_full, torch::Tensor W_piece, torch::Tensor W_rel, torch::Tensor bias,
    int64_t pad_f, bool qat, double qa_scale
) {
    c10::cuda::CUDAGuard guard(idx.device());
    int64_t B = idx.size(0);
    int K = (int)idx.size(1);
    int H = (int)W_full.size(1);
    auto out = torch::empty({B, H}, W_full.options());
    auto mask = torch::empty({B, H}, torch::TensorOptions().dtype(torch::kUInt8).device(idx.device()));
    if (B == 0) return {out, mask};
    dim3 block(256);
    dim3 grid((unsigned int)B, (unsigned int)((H + 255) / 256));
    hkv3_l1_fwd_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        idx.data_ptr<int64_t>(), idx_full.data_ptr<int64_t>(),
        piece_map.data_ptr<int64_t>(), rel_map.data_ptr<int64_t>(),
        W_full.data_ptr<float>(), W_piece.data_ptr<float>(), W_rel.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), mask.data_ptr<uint8_t>(),
        B, K, H, pad_f, qat, (float)qa_scale
    );
    return {out, mask};
}

void hkv3_l1_backward(
    torch::Tensor grad_out, torch::Tensor clamp_mask,
    torch::Tensor idx, torch::Tensor idx_full,
    torch::Tensor piece_map, torch::Tensor rel_map,
    torch::Tensor grad_W_full, torch::Tensor grad_W_piece, torch::Tensor grad_W_rel,
    int64_t pad_f
) {
    c10::cuda::CUDAGuard guard(idx.device());
    int64_t B = idx.size(0);
    int K = (int)idx.size(1);
    int H = (int)grad_W_full.size(1);
    if (B == 0) return;
    dim3 block(256);
    dim3 grid((unsigned int)B, (unsigned int)((H + 255) / 256));
    hkv3_l1_bwd_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        grad_out.data_ptr<float>(), clamp_mask.data_ptr<uint8_t>(),
        idx.data_ptr<int64_t>(), idx_full.data_ptr<int64_t>(),
        piece_map.data_ptr<int64_t>(), rel_map.data_ptr<int64_t>(),
        grad_W_full.data_ptr<float>(), grad_W_piece.data_ptr<float>(), grad_W_rel.data_ptr<float>(),
        B, K, H, pad_f
    );
}
"""
        cpp_src = """
std::vector<torch::Tensor> hkv3_l1_forward(
    torch::Tensor idx, torch::Tensor idx_full,
    torch::Tensor piece_map, torch::Tensor rel_map,
    torch::Tensor W_full, torch::Tensor W_piece, torch::Tensor W_rel, torch::Tensor bias,
    int64_t pad_f, bool qat, double qa_scale
);
void hkv3_l1_backward(
    torch::Tensor grad_out, torch::Tensor clamp_mask,
    torch::Tensor idx, torch::Tensor idx_full,
    torch::Tensor piece_map, torch::Tensor rel_map,
    torch::Tensor grad_W_full, torch::Tensor grad_W_piece, torch::Tensor grad_W_rel,
    int64_t pad_f
);
"""
        ext = load_inline(
            name="hkv3_fused_l1_ext",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hkv3_l1_forward", "hkv3_l1_backward"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        _HKV3_CUDA_EXT = ext
        print("[CUDA Kernel] Đã biên dịch thành công Fused Sparse L1 Kernel (hkv3_fused_l1_ext)!", flush=True)
        return ext
    except Exception as e:
        print(f"[CUDA Kernel] Dùng PyTorch EmbeddingBag fallback ({e})", flush=True)
        return None


class _FusedHKV3L1Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ext, idx, idx_full, piece_map, rel_map, W_full, W_piece, W_rel, bias, qat):
        out, clamp_mask = ext.hkv3_l1_forward(
            idx.contiguous(), idx_full.contiguous(), piece_map, rel_map,
            W_full, W_piece, W_rel, bias, PADDING_FEATURE_IDX, bool(qat), float(QA_SCALE)
        )
        ctx.ext = ext
        ctx.save_for_backward(clamp_mask, idx, idx_full, piece_map, rel_map, W_full, W_piece, W_rel)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        clamp_mask, idx, idx_full, piece_map, rel_map, W_full, W_piece, W_rel = ctx.saved_tensors
        g_out = grad_out.contiguous().float()
        g_W_full = torch.zeros_like(W_full)
        g_W_piece = torch.zeros_like(W_piece)
        g_W_rel = torch.zeros_like(W_rel)
        ctx.ext.hkv3_l1_backward(
            g_out, clamp_mask, idx.contiguous(), idx_full.contiguous(),
            piece_map, rel_map, g_W_full, g_W_piece, g_W_rel, PADDING_FEATURE_IDX
        )
        g_bias = (g_out * clamp_mask.to(torch.float32)).sum(dim=0)
        return None, None, None, None, None, g_W_full, g_W_piece, g_W_rel, g_bias, None


class HalfKPv3BranchL1(nn.Module):
    """
    Một nửa nhánh tích lũy Tầng 1 (1024 chiều: Left hoặc Right của Pairwise Bilinear Gating).
    Thiết kế tách nhánh này cho phép chạy Tensor-Parallel tự nhiên trên Kaggle T4x2:
      - GPU 0 (cuda:0) giữ nhánh Left  (cột 0..1023)
      - GPU 1 (cuda:1) giữ nhánh Right (cột 1024..2047)
    Đặc biệt: full_dropout CHỈ áp dụng lên bảng dư `emb_full`, giữ nguyên 100% `emb_piece` và `emb_rel`
    để không bao giờ làm mất quân cờ hay làm lệch điểm vật chất trên bàn cờ!
    """

    def __init__(
        self,
        piece_map_pad: torch.Tensor,
        rel_map_pad: torch.Tensor,
        qat: bool = True,
        full_dropout: float = 0.0,
    ):
        super().__init__()
        self.qat = qat
        self.full_dropout = full_dropout
        self.register_buffer("piece_map_pad", piece_map_pad.clone())
        self.register_buffer("rel_map_pad", rel_map_pad.clone())

        self.emb_full = nn.EmbeddingBag(
            HKV3_NUM_FEATURES + 1, HKV3_L1_PAIR_SIZE, mode="sum", sparse=False, padding_idx=PADDING_FEATURE_IDX
        )
        self.emb_piece = nn.EmbeddingBag(
            HKV3_NUM_PIECE_FEATURES + 1, HKV3_L1_PAIR_SIZE, mode="sum", sparse=False, padding_idx=PADDING_PIECE_IDX
        )
        self.emb_rel = nn.EmbeddingBag(
            HKV3_NUM_REL_FEATURES + 1, HKV3_L1_PAIR_SIZE, mode="sum", sparse=False, padding_idx=PADDING_REL_IDX
        )
        self.bias = nn.Parameter(torch.full((HKV3_L1_PAIR_SIZE,), 0.15))

        with torch.no_grad():
            nn.init.normal_(self.emb_full.weight, mean=0.0, std=0.005)
            nn.init.normal_(self.emb_piece.weight, mean=0.0, std=0.025)
            nn.init.normal_(self.emb_rel.weight, mean=0.0, std=0.020)
            self.zero_padding()

    @torch.no_grad()
    def zero_padding(self):
        self.emb_full.weight[PADDING_FEATURE_IDX].zero_()
        self.emb_piece.weight[PADDING_PIECE_IDX].zero_()
        self.emb_rel.weight[PADDING_REL_IDX].zero_()

    def forward(self, us_idx: torch.Tensor, them_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Chỉ dropout trên bảng dư emb_full để ép mạng học quy luật tổng quát qua emb_piece & emb_rel
        # mà không làm mất quân cờ hay làm lệch vật chất giữa us và them!
        if self.training and self.full_dropout > 0.0:
            pad_val = PADDING_FEATURE_IDX
            drop_u = (torch.rand_like(us_idx, dtype=torch.float32) < self.full_dropout) & (us_idx < pad_val)
            drop_v = (torch.rand_like(them_idx, dtype=torch.float32) < self.full_dropout) & (them_idx < pad_val)
            us_idx_full = torch.where(drop_u, pad_val, us_idx)
            them_idx_full = torch.where(drop_v, pad_val, them_idx)
        else:
            us_idx_full = us_idx
            them_idx_full = them_idx

        ext = _get_hkv3_cuda_ext() if (us_idx.is_cuda and us_idx.shape[1] <= 64 and self.emb_full.weight.dtype == torch.float32) else None
        if ext is not None:
            both_idx = torch.cat([us_idx, them_idx], dim=0)
            both_full = torch.cat([us_idx_full, them_idx_full], dim=0) if (us_idx_full is not us_idx or them_idx_full is not them_idx) else both_idx
            both_c = _FusedHKV3L1Op.apply(
                ext, both_idx, both_full, self.piece_map_pad, self.rel_map_pad,
                self.emb_full.weight, self.emb_piece.weight, self.emb_rel.weight, self.bias, self.qat
            )
            b = us_idx.shape[0]
            return both_c[:b], both_c[b:]

        us_p = self.piece_map_pad[us_idx]
        us_r = self.rel_map_pad[us_idx]
        them_p = self.piece_map_pad[them_idx]
        them_r = self.rel_map_pad[them_idx]

        acc_us = self.emb_full(us_idx_full) + self.emb_piece(us_p) + self.emb_rel(us_r) + self.bias
        acc_them = self.emb_full(them_idx_full) + self.emb_piece(them_p) + self.emb_rel(them_r) + self.bias

        c_us = torch.clamp(acc_us, 0.0, 1.0)
        c_them = torch.clamp(acc_them, 0.0, 1.0)
        if self.qat:
            c_us = fake_quant_ste(c_us, QA_SCALE, 0.0, 127.0)
            c_them = fake_quant_ste(c_them, QA_SCALE, 0.0, 127.0)
        return c_us, c_them

    @torch.no_grad()
    def materialize_branch_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        p_idx = self.piece_map_pad[:HKV3_NUM_FEATURES]
        r_idx = self.rel_map_pad[:HKV3_NUM_FEATURES]
        w_branch = (
            self.emb_full.weight[:HKV3_NUM_FEATURES]
            + self.emb_piece.weight[p_idx]
            + self.emb_rel.weight[r_idx]
        )
        return w_branch.detach().cpu(), self.bias.detach().cpu()


class HalfKPv3Network(nn.Module):
    """
    Kiến trúc mạng nơ-ron đa nhiệm HalfKPv3 (HKV3 - Single Big Net ~88.85 MB)
    theo đúng đặc tả tại algorithms/HalfKPv3.md:
      - Tri-Scale Virtual Factorizer: W_full(22,528) + W_piece(704) + W_rel(1,815)
      - Hỗ trợ Tensor-Parallel tự nhiên trên 2 GPU (Kaggle T4x2):
        Left Branch (1024) trên cuda:0 || Right Branch (1024) trên cuda:1
      - 4-Channel Multi-Task PSQT Skip Connection: psqt_full + psqt_piece + psqt_rel
      - Pairwise Bilinear Gating: 2048 -> 1024 per perspective (khớp bit-exact >> 7 của C++)
      - Siamese Perspective Twin FC2: W_self(256x1024) & W_opp(256x1024) (khớp bit-exact >> 6 của C++)
      - Orthogonal Parity Readout Heads: eval_head (Odd, No Bias) & tac_head (Even, With Bias)
    """

    def __init__(
        self,
        qat: bool = True,
        dev0: torch.device = torch.device("cpu"),
        dev1: Optional[torch.device] = None,
        full_dropout: float = 0.0,
        use_amp: bool = True,
    ):
        super().__init__()
        self.qat = qat
        self.dev0 = dev0
        self.dev1 = dev1 if dev1 is not None else dev0
        self.use_amp = use_amp

        piece_map_pad, rel_map_pad = build_hkv3_canonical_factorizer_maps()
        self.register_buffer("piece_map_pad", piece_map_pad.to(self.dev0))
        self.register_buffer("rel_map_pad", rel_map_pad.to(self.dev0))

        # ─── 1. Tầng 1: Hai nhánh Left (0..1023) & Right (1024..2047) của Virtual Factorizer ───
        self.left_branch = HalfKPv3BranchL1(piece_map_pad, rel_map_pad, qat=qat, full_dropout=full_dropout).to(self.dev0)
        self.right_branch = HalfKPv3BranchL1(piece_map_pad, rel_map_pad, qat=qat, full_dropout=full_dropout).to(self.dev1)

        # ─── 2. Nhánh tắt PSQT 4 Kênh (trên dev0) ───
        self.psqt_full = nn.EmbeddingBag(
            HKV3_NUM_FEATURES + 1, HKV3_PSQT_BUCKETS, mode="sum", sparse=False, padding_idx=PADDING_FEATURE_IDX
        ).to(self.dev0)
        self.psqt_piece = nn.EmbeddingBag(
            HKV3_NUM_PIECE_FEATURES + 1, HKV3_PSQT_BUCKETS, mode="sum", sparse=False, padding_idx=PADDING_PIECE_IDX
        ).to(self.dev0)
        self.psqt_rel = nn.EmbeddingBag(
            HKV3_NUM_REL_FEATURES + 1, HKV3_PSQT_BUCKETS, mode="sum", sparse=False, padding_idx=PADDING_REL_IDX
        ).to(self.dev0)

        # ─── 3. Tầng 2: Siamese Perspective Twin FC2 (W_self và W_opp trên dev0) ───
        self.fc2_self = nn.Linear(HKV3_L1_PAIR_SIZE, HKV3_L2_SIZE, bias=True).to(self.dev0)
        self.fc2_opp = nn.Linear(HKV3_L1_PAIR_SIZE, HKV3_L2_SIZE, bias=False).to(self.dev0)

        # ─── 4. Tầng 3: Đầu đọc Trực giao Chẵn - Lẻ (trên dev0) ───
        self.eval_head = nn.Linear(HKV3_L2_SIZE, 1, bias=False).to(self.dev0)
        self.tac_head = nn.Linear(HKV3_L2_SIZE, 3, bias=True).to(self.dev0)

        self._init_head_and_psqt_weights()

    def set_full_dropout(self, p: float):
        self.left_branch.full_dropout = p
        self.right_branch.full_dropout = p

    def _init_head_and_psqt_weights(self):
        with torch.no_grad():
            nn.init.zeros_(self.psqt_full.weight)
            nn.init.normal_(self.psqt_piece.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.psqt_rel.weight)
            self.psqt_full.weight[PADDING_FEATURE_IDX].zero_()
            self.psqt_piece.weight[PADDING_PIECE_IDX].zero_()
            self.psqt_rel.weight[PADDING_REL_IDX].zero_()

            # Khởi tạo giá trị vật chất cơ bản (P=1, N=3, B=3.1, R=5, Q=9 pawns) vào psqt_piece kênh 0
            piece_vals = [1.0, 3.0, 3.1, 5.0, 9.0, -1.0, -3.0, -3.1, -5.0, -9.0, 0.0]
            for c, val in enumerate(piece_vals):
                if c < 10:
                    self.psqt_piece.weight[c * 64:(c + 1) * 64, 0] = val * 0.5

            nn.init.kaiming_normal_(self.fc2_self.weight, nonlinearity="relu")
            self.fc2_self.weight.mul_(0.35)
            nn.init.zeros_(self.fc2_self.bias)

            nn.init.kaiming_normal_(self.fc2_opp.weight, nonlinearity="relu")
            self.fc2_opp.weight.mul_(0.35)

            nn.init.xavier_uniform_(self.eval_head.weight, gain=0.5)
            nn.init.xavier_uniform_(self.tac_head.weight, gain=0.5)
            nn.init.zeros_(self.tac_head.bias)

    @torch.no_grad()
    def clamp_quantizable_weights(self):
        max_fc2 = 127.0 / QW_FC2_SCALE
        self.fc2_self.weight.clamp_(-max_fc2, max_fc2)
        self.fc2_opp.weight.clamp_(-max_fc2, max_fc2)

        max_head = 32767.0 / Q_HEAD_SCALE
        self.eval_head.weight.clamp_(-max_head, max_head)
        self.tac_head.weight.clamp_(-max_head, max_head)

        self.left_branch.zero_padding()
        self.right_branch.zero_padding()
        self.psqt_full.weight[PADDING_FEATURE_IDX].zero_()
        self.psqt_piece.weight[PADDING_PIECE_IDX].zero_()
        self.psqt_rel.weight[PADDING_REL_IDX].zero_()

    def _compute_psqt(self, idx: torch.Tensor) -> torch.Tensor:
        p_idx = self.piece_map_pad[idx]
        r_idx = self.rel_map_pad[idx]
        return self.psqt_full(idx) + self.psqt_piece(p_idx) + self.psqt_rel(r_idx)

    def forward(self, us_idx: torch.Tensor, them_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Bước 1: Chạy song song nhánh Left trên dev0 và nhánh Right trên dev1 (Tensor-Parallel trên T4x2!)
        if self.dev1 != self.dev0:
            amp_enabled = us_idx.is_cuda and self.use_amp
            us_idx_1 = us_idx.to(self.dev1, non_blocking=True)
            them_idx_1 = them_idx.to(self.dev1, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                u_right, v_right = self.right_branch(us_idx_1, them_idx_1)
            u_left, v_left = self.left_branch(us_idx, them_idx)
            # Nén sang FP16 trên dev1 trước khi truyền qua PCIe về dev0 để giảm 50% băng thông PCIe:
            if amp_enabled:
                u_right = u_right.to(dtype=torch.float16).to(self.dev0, non_blocking=True).to(dtype=u_left.dtype)
                v_right = v_right.to(dtype=torch.float16).to(self.dev0, non_blocking=True).to(dtype=v_left.dtype)
            else:
                u_right = u_right.to(self.dev0, non_blocking=True)
                v_right = v_right.to(self.dev0, non_blocking=True)
        else:
            u_left, v_left = self.left_branch(us_idx, them_idx)
            u_right, v_right = self.right_branch(us_idx, them_idx)

        # Bước 2: Pairwise Bilinear Gating khớp chính xác phép dịch >> 7 (chia 128 floor) của C++ AVX2
        u = bilinear_gate_ste(u_left, u_right, self.qat)
        v = bilinear_gate_ste(v_left, v_right, self.qat)

        psqt_us = self._compute_psqt(us_idx)
        psqt_them = self._compute_psqt(them_idx)

        # Bước 3: Siamese Perspective Twin FC2 (QAT luôn bật cả lúc train và eval để val khớp 100% với .harenn)
        if self.qat:
            w_self = fake_quant_ste(self.fc2_self.weight, QW_FC2_SCALE, -127.0, 127.0)
            b_fc2 = fake_quant_ste(self.fc2_self.bias, QA_SCALE * QW_FC2_SCALE, -2147483647.0, 2147483647.0)
            w_opp = fake_quant_ste(self.fc2_opp.weight, QW_FC2_SCALE, -127.0, 127.0)
            z_us = F.linear(u, w_self, b_fc2) + F.linear(v, w_opp, None)
            z_them = F.linear(v, w_self, b_fc2) + F.linear(u, w_opp, None)
        else:
            z_us = self.fc2_self(u) + self.fc2_opp(v)
            z_them = self.fc2_self(v) + self.fc2_opp(u)

        # Bước 4: Kích hoạt phi tuyến SCReLU tại tầng 2 (khớp hoàn toàn với z >> 6 floor và clamp 0..127 trong C++)
        h_us = fc2_screlu_ste(z_us, self.qat)
        h_them = fc2_screlu_ste(z_them, self.qat)

        # Bước 5: Phân rã Trực giao Chẵn - Lẻ ngay sau phi tuyến FC2
        h_odd = h_us - h_them
        h_even = h_us + h_them

        # Bước 6: Đầu đọc + Nhánh tắt PSQT
        psqt_eval = psqt_us[:, 0] - psqt_them[:, 0]
        psqt_tac = psqt_us[:, 1:4] + psqt_them[:, 1:4]

        if self.qat:
            eval_w = fake_quant_ste(self.eval_head.weight, Q_HEAD_SCALE, -32767.0, 32767.0)
            tac_w = fake_quant_ste(self.tac_head.weight, Q_HEAD_SCALE, -32767.0, 32767.0)
            tac_b = fake_quant_ste(self.tac_head.bias, OUTPUT_SCALE, -2147483647.0, 2147483647.0)
            eval_pawns = F.linear(h_odd, eval_w, None).squeeze(-1) + psqt_eval
            tac_logits = F.linear(h_even, tac_w, tac_b) + psqt_tac
        else:
            eval_pawns = self.eval_head(h_odd).squeeze(-1) + psqt_eval
            tac_logits = self.tac_head(h_even) + psqt_tac

        return {
            "eval": eval_pawns,
            "tau": tac_logits[:, 0],
            "rho": tac_logits[:, 1],
            "rs": tac_logits[:, 2],
        }

    @torch.no_grad()
    def materialize_l1_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gộp phẳng 3 không gian trọng số của hai nhánh Left (0..1023) & Right (1024..2047)
        thành ma trận duy nhất [22,528 x 2,048], bias [2,048] và PSQT [22,528 x 4].
        """
        w_left, b_left = self.left_branch.materialize_branch_weights()
        w_right, b_right = self.right_branch.materialize_branch_weights()
        w_eff = torch.cat([w_left, w_right], dim=-1)
        b_eff = torch.cat([b_left, b_right], dim=-1)

        p_idx = self.piece_map_pad[:HKV3_NUM_FEATURES]
        r_idx = self.rel_map_pad[:HKV3_NUM_FEATURES]
        psqt_eff = (
            self.psqt_full.weight[:HKV3_NUM_FEATURES]
            + self.psqt_piece.weight[p_idx]
            + self.psqt_rel.weight[r_idx]
        ).detach().cpu()
        return w_eff, b_eff, psqt_eff


# ═══════════════════════════════════════════════════════════════════════════
# 6. HÀM MẤT MÁT ĐA NHIỆM (WDL BCE + HUBER CENTIPAWN + AUXILIARY HEADS)
# ═══════════════════════════════════════════════════════════════════════════
class HKV3MultiTaskLoss(nn.Module):
    """
    Hàm mục tiêu huấn luyện đa nhiệm chuẩn Stockfish NNUE + HARENN:
      - Eval Loss: Kết hợp WDL Cross-Entropy (trên xác suất thắng/hòa/thua) và Smooth-L1 Huber Loss (trên Pawns).
      - Auxiliary Search Loss: Binary Cross-Entropy trên logits của tau, rho, rs.
      - L2 Regularization nhẹ trên bảng dư W_full để khuyến khích mạng ưu tiên học quy luật hình học tổng quát ở W_rel & W_piece.
    """

    def __init__(self, wdl_weight: float = 0.75, wdl_lambda: float = 0.80, wdl_scale_pawns: float = WDL_SCALE_PAWNS):
        super().__init__()
        self.wdl_weight = wdl_weight
        self.wdl_lambda = wdl_lambda
        self.wdl_scale_pawns = wdl_scale_pawns

    def compute_eval_loss(
        self,
        pred_pawns: torch.Tensor,
        target_pawns: torch.Tensor,
        target_wdl_game: Optional[torch.Tensor] = None,
        eval_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_clamped = torch.clamp(target_pawns, -10.0, 10.0)

        # 1. Smooth-L1 Huber Loss trực tiếp trên đơn vị Pawns (beta = 0.5 pawns = 50 cp)
        loss_huber = F.smooth_l1_loss(pred_pawns, target_clamped, reduction="none", beta=0.5)

        # 2. WDL Sigmoid Binary Cross-Entropy: trộn eval teacher với kết quả ván thật (lambda blending)
        wdl_eval = torch.sigmoid(target_clamped / self.wdl_scale_pawns)
        if target_wdl_game is not None and self.wdl_lambda < 1.0:
            target_wdl = self.wdl_lambda * wdl_eval + (1.0 - self.wdl_lambda) * target_wdl_game
        else:
            target_wdl = wdl_eval
        loss_wdl = F.binary_cross_entropy_with_logits(
            pred_pawns / self.wdl_scale_pawns, target_wdl, reduction="none"
        )

        per_pos = self.wdl_weight * loss_wdl + (1.0 - self.wdl_weight) * loss_huber
        if eval_mask is not None:
            denom = eval_mask.sum().clamp(min=1.0)
            return (per_pos * eval_mask).sum() / denom
        return per_pos.mean()

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        eval_mask = targets.get("eval_mask")
        loss_eval = self.compute_eval_loss(
            outputs["eval"],
            targets["eval"],
            target_wdl_game=targets.get("wdl"),
            eval_mask=eval_mask,
        )

        loss_tau = F.binary_cross_entropy_with_logits(outputs["tau"], targets["tau"], reduction="mean")
        loss_rho = F.binary_cross_entropy_with_logits(outputs["rho"], targets["rho"], reduction="mean")
        loss_rs = F.binary_cross_entropy_with_logits(outputs["rs"], targets["rs"], reduction="mean")

        total_loss = loss_eval + 0.15 * (loss_tau + loss_rho + loss_rs)

        with torch.no_grad():
            pred_wdl = torch.sigmoid(outputs["eval"] / self.wdl_scale_pawns)
            tgt_wdl = torch.sigmoid(targets["eval"] / self.wdl_scale_pawns)
            abs_err = torch.abs(outputs["eval"] - targets["eval"])
            w_err = torch.abs(pred_wdl - tgt_wdl) ** 2.5
            if eval_mask is not None:
                denom = eval_mask.sum().clamp(min=1.0)
                mae_cp = ((abs_err * eval_mask).sum() / denom).item() * 100.0
                wdl_err = ((w_err * eval_mask).sum() / denom).item()
            else:
                mae_cp = torch.mean(abs_err).item() * 100.0
                wdl_err = torch.mean(w_err).item()

        metrics = {
            "total": float(total_loss.item()),
            "eval": float(loss_eval.item()),
            "tau": float(loss_tau.item()),
            "rho": float(loss_rho.item()),
            "rs": float(loss_rs.item()),
            "mae_cp": mae_cp,
            "wdl_err": wdl_err,
        }
        return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
# 7. XUẤT FILE NHỊ PHÂN .HARENN & KIỂM CHỨNG ĐỐI XỨNG SỐ NGUYÊN C++ AVX2
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def export_hkv3_binary(model: HalfKPv3Network, filepath: str):
    """
    Xuất mô hình HalfKPv3 ra file nhị phân .harenn (~88.85 MB) theo đúng bố cục bộ nhớ
    của struct HKV3Weights trong algorithms/HalfKPv3.md.
    """
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()
    raw_model.clamp_quantizable_weights()

    w_eff, b_eff, psqt_eff = raw_model.materialize_l1_weights()

    # 1. Lượng tử hóa Tầng 1 (22,528 x 2,048 int16 = 88.00 MB)
    fc1_w_i16 = (
        torch.clamp(torch.round(w_eff * QA_SCALE), -32767, 32767)
        .to(torch.int16)
        .cpu()
        .numpy()
    )
    fc1_b_i16 = (
        torch.clamp(torch.round(b_eff * QA_SCALE), -32767, 32767)
        .to(torch.int16)
        .cpu()
        .numpy()
    )
    psqt_i32 = (
        torch.clamp(torch.round(psqt_eff * OUTPUT_SCALE), -2147483647, 2147483647)
        .to(torch.int32)
        .cpu()
        .numpy()
    )

    # 2. Lượng tử hóa Tầng 2 Siamese Perspective Twin FC2 (int8)
    w_self_i8 = (
        torch.clamp(torch.round(raw_model.fc2_self.weight * QW_FC2_SCALE), -127, 127)
        .to(torch.int8)
        .cpu()
        .numpy()
    )
    w_opp_i8 = (
        torch.clamp(torch.round(raw_model.fc2_opp.weight * QW_FC2_SCALE), -127, 127)
        .to(torch.int8)
        .cpu()
        .numpy()
    )

    b_fc2_i32 = (
        torch.clamp(
            torch.round(raw_model.fc2_self.bias * (QA_SCALE * QW_FC2_SCALE)),
            -2147483647,
            2147483647,
        )
        .to(torch.int32)
        .cpu()
        .numpy()
    )

    # 3. Lượng tử hóa Tầng 3 Đầu đọc Chẵn - Lẻ
    eval_w_i16 = (
        torch.clamp(torch.round(raw_model.eval_head.weight.squeeze(0) * Q_HEAD_SCALE), -32767, 32767)
        .to(torch.int16)
        .cpu()
        .numpy()
    )
    tac_w_i16 = (
        torch.clamp(torch.round(raw_model.tac_head.weight * Q_HEAD_SCALE), -32767, 32767)
        .to(torch.int16)
        .cpu()
        .numpy()
    )
    tac_b_i32 = (
        torch.clamp(torch.round(raw_model.tac_head.bias * OUTPUT_SCALE), -2147483647, 2147483647)
        .to(torch.int32)
        .cpu()
        .numpy()
    )

    out_path = Path(filepath)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(HKV3_MAGIC)
        f.write(struct.pack("<I", HKV3_ARCH_ID))
        f.write(fc1_w_i16.tobytes())
        f.write(fc1_b_i16.tobytes())
        f.write(psqt_i32.tobytes())
        f.write(w_self_i8.tobytes())
        f.write(w_opp_i8.tobytes())
        f.write(b_fc2_i32.tobytes())
        f.write(eval_w_i16.tobytes())
        f.write(tac_w_i16.tobytes())
        f.write(tac_b_i32.tobytes())

    size_bytes = out_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"[Export] Đã xuất thành công file nhị phân HKV3: {out_path} ({size_bytes:,} bytes = {size_mb:.2f} MB)", flush=True)
    return {
        "fc1_w": fc1_w_i16,
        "fc1_b": fc1_b_i16,
        "psqt": psqt_i32,
        "w_self": w_self_i8,
        "w_opp": w_opp_i8,
        "b_fc2": b_fc2_i32,
        "eval_w": eval_w_i16,
        "tac_w": tac_w_i16,
        "tac_b": tac_b_i32,
    }


def verify_integer_avx2_parity_and_accuracy(
    quant_tensors: dict,
    us_sample: np.ndarray,
    them_sample: np.ndarray,
    model: Optional[HalfKPv3Network] = None,
):
    """
    Mô phỏng chính xác 100% đường đi số nguyên C++ AVX2 (_mm256_maddubs_epi16) trên tập mẫu để:
      1. Kiểm chứng không có tràn số bão hòa int16_t trong maddubs (u_prod, v_prod in [0, 127]).
      2. Kiểm chứng định lý Phản đối xứng BIT-EXACT: E_int(us, them) + E_int(them, us) == 0.
      3. Kiểm chứng định lý Đối xứng BIT-EXACT: [tau, rho, rs]_int(us, them) == [tau, rho, rs]_int(them, us).
      4. Đo sai số lượng tử hóa thực tế (MAE & MaxDiff tính bằng centipawn) giữa mạng PyTorch QAT và C++ AVX2 Integer.
    """
    fc1_w = quant_tensors["fc1_w"]
    fc1_b = quant_tensors["fc1_b"]
    psqt = quant_tensors["psqt"]
    w_self = quant_tensors["w_self"].astype(np.int32)
    w_opp = quant_tensors["w_opp"].astype(np.int32)
    b_fc2 = quant_tensors["b_fc2"].astype(np.int32)
    eval_w = quant_tensors["eval_w"].astype(np.int64)
    tac_w = quant_tensors["tac_w"].astype(np.int64)
    tac_b = quant_tensors["tac_b"].astype(np.int64)

    def run_int_inference(us_arr, them_arr):
        n = len(us_arr)
        acc_us = np.tile(fc1_b.astype(np.int32), (n, 1))
        acc_them = np.tile(fc1_b.astype(np.int32), (n, 1))
        psqt_us = np.zeros((n, 4), dtype=np.int64)
        psqt_them = np.zeros((n, 4), dtype=np.int64)

        for i in range(n):
            u_valid = us_arr[i][us_arr[i] < HKV3_NUM_FEATURES]
            v_valid = them_arr[i][them_arr[i] < HKV3_NUM_FEATURES]
            acc_us[i] += np.sum(fc1_w[u_valid].astype(np.int32), axis=0)
            acc_them[i] += np.sum(fc1_w[v_valid].astype(np.int32), axis=0)
            psqt_us[i] = np.sum(psqt[u_valid].astype(np.int64), axis=0)
            psqt_them[i] = np.sum(psqt[v_valid].astype(np.int64), axis=0)

        # Pairwise Bilinear Gating (int16 clamp [0, 127], mullo, srli 7 -> uint8_t in [0, 127])
        u_clip = np.clip(acc_us, 0, 127)
        v_clip = np.clip(acc_them, 0, 127)
        u_u8 = (u_clip[:, :1024] * u_clip[:, 1024:]) >> 7  # [N, 1024] in [0, 127]
        v_u8 = (v_clip[:, :1024] * v_clip[:, 1024:]) >> 7  # [N, 1024] in [0, 127]

        assert np.all((u_u8 >= 0) & (u_u8 <= 127)), "u_u8 out of [0, 127]!"
        assert np.all((v_u8 >= 0) & (v_u8 <= 127)), "v_u8 out of [0, 127]!"

        # Siamese Perspective Twin FC2:
        #   z_us   = (W_self * u_u8 + W_opp * v_u8 + b_fc2) >> 6
        #   z_them = (W_self * v_u8 + W_opp * u_u8 + b_fc2) >> 6
        z_us = ((u_u8 @ w_self.T) + (v_u8 @ w_opp.T) + b_fc2[None, :]) >> FC2_SHIFT
        z_them = ((v_u8 @ w_self.T) + (u_u8 @ w_opp.T) + b_fc2[None, :]) >> FC2_SHIFT

        h_us = np.clip(z_us, 0, 127) ** 2
        h_them = np.clip(z_them, 0, 127) ** 2

        h_odd = (h_us - h_them).astype(np.int64)
        h_even = (h_us + h_them).astype(np.int64)

        eval_raw = (h_odd @ eval_w) + (psqt_us[:, 0] - psqt_them[:, 0])
        tac_raw = (h_even @ tac_w.T) + tac_b[None, :] + (psqt_us[:, 1:4] + psqt_them[:, 1:4])
        return eval_raw, tac_raw

    eval_fwd, tac_fwd = run_int_inference(us_sample, them_sample)
    eval_rev, tac_rev = run_int_inference(them_sample, us_sample)

    max_eval_asym = int(np.max(np.abs(eval_fwd + eval_rev)))
    max_tac_asym = int(np.max(np.abs(tac_fwd - tac_rev)))
    assert max_eval_asym == 0, f"Eval parity violation: {max_eval_asym}"
    assert max_tac_asym == 0, f"Tactical parity violation: {max_tac_asym}"
    print(f"[Verify AVX2] Kiểm chứng bit-exact trên {len(us_sample)} thế cờ thực tế: "
          f"Eval Parity Error = {max_eval_asym} (0.0000 cp) | Tactical Parity Error = {max_tac_asym} (0.0000) -> PASS 100%!", flush=True)

    if model is not None:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            us_t = torch.from_numpy(us_sample.astype(np.int64)).to(model.dev0)
            them_t = torch.from_numpy(them_sample.astype(np.int64)).to(model.dev0)
            out = model(us_t, them_t)
            eval_float_cp = out["eval"].detach().cpu().numpy().astype(np.float64) * 100.0
            tac_float = np.stack([
                out["tau"].detach().cpu().numpy(),
                out["rho"].detach().cpu().numpy(),
                out["rs"].detach().cpu().numpy(),
            ], axis=1).astype(np.float64)
        if was_training:
            model.train()

        eval_int_cp = (eval_fwd.astype(np.float64) / OUTPUT_SCALE) * 100.0
        tac_int = tac_fwd.astype(np.float64) / OUTPUT_SCALE
        eval_diff = np.abs(eval_float_cp - eval_int_cp)
        tac_diff = np.abs(tac_float - tac_int)
        print(
            f"[Verify QAT vs C++ AVX2] Sai số lượng tử hóa PyTorch <-> C++ Integer: "
            f"Eval MAE = {np.mean(eval_diff):.5f} cp (MaxDiff = {np.max(eval_diff):.5f} cp) | "
            f"Tactical Logit MAE = {np.mean(tac_diff):.6f}",
            flush=True,
        )


def _decompose_triscale_table(
    flat_w: torch.Tensor,
    rel_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phân rã trực giao ngược bảng trọng số phẳng [22528, D] về 3 tầng của Tri-Scale Virtual Factorizer:
      1. W_piece [704, D]   = Trung bình cộng qua 32 ô Vua (giữ tri thức giá trị vật chất & PSQT toàn cục).
      2. W_rel   [2475, D]  = Trung bình cộng phần dư theo từng khoảng cách tương đối với Vua (c, dx, dy).
      3. W_full  [22528, D] = Phần dư đặc thù riêng của từng ô Vua.
    Đảm bảo: W_full + W_piece[piece_map] + W_rel[rel_map] == flat_w chính xác tuyệt đối,
    giúp khi resume từ file .harenn không bị weight_decay của W_full xóa mất tri thức vật chất.
    """
    d = flat_w.shape[1]
    w_3d = flat_w.view(HKV3_KING_BUCKETS, HKV3_NUM_PIECE_FEATURES, d)
    w_piece = w_3d.mean(dim=0)  # [704, D]
    res1 = flat_w - w_piece.repeat(HKV3_KING_BUCKETS, 1)  # [22528, D]

    rel_idx = rel_map.to(flat_w.device)
    rel_sum = torch.zeros((HKV3_NUM_REL_FEATURES, d), dtype=flat_w.dtype, device=flat_w.device)
    rel_cnt = torch.zeros((HKV3_NUM_REL_FEATURES, 1), dtype=flat_w.dtype, device=flat_w.device)
    rel_sum.index_add_(0, rel_idx, res1)
    ones = torch.ones((HKV3_NUM_FEATURES, 1), dtype=flat_w.dtype, device=flat_w.device)
    rel_cnt.index_add_(0, rel_idx, ones)
    w_rel = rel_sum / torch.clamp(rel_cnt, min=1.0)  # [2475, D]

    w_full = res1 - w_rel[rel_idx]  # [22528, D]
    return w_full, w_piece, w_rel


def load_hkv3_weights_or_checkpoint(
    model: HalfKPv3Network,
    filepath: str,
    dev0: torch.device,
    dev1: torch.device,
) -> Tuple[dict, int]:
    """
    Nạp trọng số từ checkpoint .pt hoặc file nhị phân .harenn cũ.
    Hỗ trợ 2 kịch bản:
      1. Checkpoint PyTorch (.pt): Khôi phục nguyên vẹn weights, optimizer, scaler, scheduler.
      2. File nhị phân (.harenn): Khôi phục weights bit-exact 100% đồng thời phân rã ngược về
         3 bảng (W_full, W_piece, W_rel), cho phép resume/finetune ngay cả khi không giữ file .pt.
    Trả về: (checkpoint_dict, loaded_epoch)
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint/model: {filepath}")

    with open(p, "rb") as f:
        header = f.read(4)

    if header == HKV3_MAGIC:
        print(f"[Resume] Phát hiện file nhị phân HKV3 (.harenn): {filepath}", flush=True)
        with open(p, "rb") as f:
            magic = f.read(4)
            arch_id = struct.unpack("<I", f.read(4))[0]
            if magic != HKV3_MAGIC or arch_id != HKV3_ARCH_ID:
                raise ValueError(f"File {filepath} không đúng định dạng HKV3 (magic={magic}, arch={arch_id})")

            fc1_w = np.frombuffer(f.read(HKV3_NUM_FEATURES * HKV3_L1_SIZE * 2), dtype=np.int16).reshape(HKV3_NUM_FEATURES, HKV3_L1_SIZE)
            fc1_b = np.frombuffer(f.read(HKV3_L1_SIZE * 2), dtype=np.int16)
            psqt = np.frombuffer(f.read(HKV3_NUM_FEATURES * HKV3_PSQT_BUCKETS * 4), dtype=np.int32).reshape(HKV3_NUM_FEATURES, HKV3_PSQT_BUCKETS)
            w_self = np.frombuffer(f.read(HKV3_L2_SIZE * HKV3_L1_PAIR_SIZE), dtype=np.int8).reshape(HKV3_L2_SIZE, HKV3_L1_PAIR_SIZE)
            w_opp = np.frombuffer(f.read(HKV3_L2_SIZE * HKV3_L1_PAIR_SIZE), dtype=np.int8).reshape(HKV3_L2_SIZE, HKV3_L1_PAIR_SIZE)
            b_fc2 = np.frombuffer(f.read(HKV3_L2_SIZE * 4), dtype=np.int32)
            eval_w = np.frombuffer(f.read(HKV3_L2_SIZE * 2), dtype=np.int16)
            tac_w = np.frombuffer(f.read(3 * HKV3_L2_SIZE * 2), dtype=np.int16).reshape(3, HKV3_L2_SIZE)
            tac_b = np.frombuffer(f.read(3 * 4), dtype=np.int32)

        with torch.no_grad():
            # Phân rã ngược L1 về 3 tầng (W_full, W_piece, W_rel) trên cả 2 nhánh Left & Right
            left_flat = torch.from_numpy(fc1_w[:, :HKV3_L1_PAIR_SIZE].astype(np.float32) / QA_SCALE).to(dev0)
            l_full, l_piece, l_rel = _decompose_triscale_table(
                left_flat, model.left_branch.rel_map_pad[:HKV3_NUM_FEATURES]
            )
            model.left_branch.emb_full.weight.data[:HKV3_NUM_FEATURES] = l_full
            model.left_branch.emb_full.weight.data[HKV3_NUM_FEATURES:].zero_()
            model.left_branch.emb_piece.weight.data[:HKV3_NUM_PIECE_FEATURES] = l_piece
            model.left_branch.emb_piece.weight.data[HKV3_NUM_PIECE_FEATURES:].zero_()
            model.left_branch.emb_rel.weight.data[:HKV3_NUM_REL_FEATURES] = l_rel
            model.left_branch.emb_rel.weight.data[HKV3_NUM_REL_FEATURES:].zero_()
            model.left_branch.bias.data = torch.from_numpy(fc1_b[:HKV3_L1_PAIR_SIZE].astype(np.float32) / QA_SCALE).to(dev0)

            right_flat = torch.from_numpy(fc1_w[:, HKV3_L1_PAIR_SIZE:].astype(np.float32) / QA_SCALE).to(dev1)
            r_full, r_piece, r_rel = _decompose_triscale_table(
                right_flat, model.right_branch.rel_map_pad[:HKV3_NUM_FEATURES]
            )
            model.right_branch.emb_full.weight.data[:HKV3_NUM_FEATURES] = r_full
            model.right_branch.emb_full.weight.data[HKV3_NUM_FEATURES:].zero_()
            model.right_branch.emb_piece.weight.data[:HKV3_NUM_PIECE_FEATURES] = r_piece
            model.right_branch.emb_piece.weight.data[HKV3_NUM_PIECE_FEATURES:].zero_()
            model.right_branch.emb_rel.weight.data[:HKV3_NUM_REL_FEATURES] = r_rel
            model.right_branch.emb_rel.weight.data[HKV3_NUM_REL_FEATURES:].zero_()
            model.right_branch.bias.data = torch.from_numpy(fc1_b[HKV3_L1_PAIR_SIZE:].astype(np.float32) / QA_SCALE).to(dev1)

            # Phân rã ngược PSQT về 3 tầng (psqt_full, psqt_piece, psqt_rel)
            psqt_flat = torch.from_numpy(psqt.astype(np.float32) / OUTPUT_SCALE).to(dev0)
            p_full, p_piece, p_rel = _decompose_triscale_table(
                psqt_flat, model.rel_map_pad[:HKV3_NUM_FEATURES]
            )
            model.psqt_full.weight.data[:HKV3_NUM_FEATURES] = p_full
            model.psqt_full.weight.data[HKV3_NUM_FEATURES:].zero_()
            model.psqt_piece.weight.data[:HKV3_NUM_PIECE_FEATURES] = p_piece
            model.psqt_piece.weight.data[HKV3_NUM_PIECE_FEATURES:].zero_()
            model.psqt_rel.weight.data[:HKV3_NUM_REL_FEATURES] = p_rel
            model.psqt_rel.weight.data[HKV3_NUM_REL_FEATURES:].zero_()

            # Nạp FC2
            model.fc2_self.weight.data = torch.from_numpy(w_self.astype(np.float32) / QW_FC2_SCALE).to(dev0)
            model.fc2_opp.weight.data = torch.from_numpy(w_opp.astype(np.float32) / QW_FC2_SCALE).to(dev0)
            model.fc2_self.bias.data = torch.from_numpy(b_fc2.astype(np.float32) / (QA_SCALE * QW_FC2_SCALE)).to(dev0)

            # Nạp Heads
            model.eval_head.weight.data = torch.from_numpy(eval_w.astype(np.float32) / Q_HEAD_SCALE).unsqueeze(0).to(dev0)
            model.tac_head.weight.data = torch.from_numpy(tac_w.astype(np.float32) / Q_HEAD_SCALE).to(dev0)
            model.tac_head.bias.data = torch.from_numpy(tac_b.astype(np.float32) / OUTPUT_SCALE).to(dev0)

            model.clamp_quantizable_weights()

        print(f"[Resume] Nạp & phân rã Tri-Scale (W_full + W_piece + W_rel) bit-exact từ {filepath} thành công!", flush=True)

        # Tự động trích xuất số epoch nếu có trong tên file .harenn (vd: nextfish_epoch4.harenn, net_ep08.harenn)
        match = re.search(r'(?:epoch|ep)[_-]?(\d+)', p.name, re.IGNORECASE)
        inferred_epoch = int(match.group(1)) if match else 0
        if inferred_epoch > 0:
            print(f"[Resume] Đã tự động nhận diện Epoch {inferred_epoch} từ tên file: {p.name}", flush=True)

        return {"type": "harenn", "epoch": inferred_epoch}, inferred_epoch

    # Ngược lại: file checkpoint PyTorch (.pt / .pth)
    print(f"[Resume] Đang nạp checkpoint PyTorch từ {filepath}...", flush=True)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        loaded_epoch = int(ckpt.get("epoch", 0))
    else:
        state = ckpt
        loaded_epoch = 0

    model.load_state_dict(state, strict=False)
    model.left_branch.to(dev0)
    model.right_branch.to(dev1)
    model.clamp_quantizable_weights()
    print(f"[Resume] Nạp checkpoint PyTorch thành công! (Checkpoint Epoch: {loaded_epoch})", flush=True)
    return (ckpt if isinstance(ckpt, dict) else {}), loaded_epoch


# ═══════════════════════════════════════════════════════════════════════════
# 8. VÒNG LẶP HUẤN LUYỆN CHÍNH (HỖ TRỢ ĐA GPU KAGGLE T4x2 + AMP FP16)
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="HalfKPv3 (HKV3) + HARENN Training Pipeline (Kaggle T4x2 Ready)"
    )
    parser.add_argument("--data", type=str, default="Neuron Network/data", help="Thư mục chứa các file .jsonl/.json hoặc .npz")
    parser.add_argument("--cache", type=str, default="", help="Đường dẫn file cache .npz đơn lẻ (tùy chọn)")
    parser.add_argument("--cache-dir", type=str, default="", help="Thư mục chứa nhiều file .npz để train streaming (tìm đệ quy, vd: /kaggle/input)")
    parser.add_argument("--kaggle-index", type=str, default="", help="Kaggle Index Dataset (vd: hunganh12333/harenn-hkv3-npz-index) để tự tải luân phiên từng part Public")
    parser.add_argument("--chunk-max-pos", type=int, default=80_000_000, help="Số thế cờ tối đa nạp RAM mỗi super-chunk trong chế độ --cache-dir (mặc định: 80M ≈ 12 GB RAM)")
    parser.add_argument("--start-part", type=int, default=0, help="Lọc bắt đầu từ part >= N khi dùng --kaggle-index (0 = tất cả)")
    parser.add_argument("--end-part", type=int, default=0, help="Lọc kết thúc ở part <= N khi dùng --kaggle-index (0 = tất cả)")
    parser.add_argument("--time-budget", type=int, default=0, help="Giới hạn thời gian train (giây, 0 = không giới hạn; trên Kaggle nên đặt 39600)")
    parser.add_argument("--hf-log-repo", type=str, default="", help="HF Dataset repo đẩy live status/log & backup .harenn (vd: australia1/harenn-hkv3-logs)")
    parser.add_argument("--live-log-file", type=str, default="", help="Đường dẫn file log để đồng bộ lên --hf-log-repo")
    parser.add_argument("--sync-hf", action="store_true", help="Tự động tải toàn bộ .jsonl từ Hugging Face Bucket trước khi train")
    parser.add_argument("--hf-bucket", type=str, default="australia1/myrandombucket", help="ID của Hugging Face Bucket")
    parser.add_argument("--epochs", type=int, default=16, help="Số epoch huấn luyện đa nhiệm Giai đoạn 1")
    parser.add_argument("--finetune-eval-epochs", type=int, default=4, help="Số epoch tinh chỉnh chuyên sâu Eval ở Giai đoạn 2")
    parser.add_argument("--batch-size", type=int, default=16384, help="Kích thước batch (khuyến nghị 16384 cho T4x2)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate cơ sở cho AdamW")
    parser.add_argument("--wdl-weight", type=float, default=0.75, help="Tỷ trọng WDL BCE Loss so với Huber Loss (mặc định: 0.75)")
    parser.add_argument("--wdl-lambda", type=float, default=0.80, help="Tỷ trọng eval teacher so với kết quả ván thật wdl (mặc định: 0.80)")
    parser.add_argument("--symmetry", type=int, default=1, choices=[1, 2], help="Chế độ đối xứng lật màu dọc (mặc định: 1x do kiến trúc đã đối xứng đại số tuyệt đối)")
    parser.add_argument("--feature-dropout", type=float, default=0.0, help="Tỷ lệ dropout chỉ riêng trên bảng W_full ở Stage 1 (mặc định: 0.0 - tắt hoàn toàn)")
    parser.add_argument("--val-split", type=float, default=0.03, help="Tỷ lệ tập validation")
    parser.add_argument("--no-qat", action="store_true", help="Tắt Quantization-Aware Training")
    parser.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision (FP16 Tensor Cores)")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn checkpoint (.pt) hoặc file nhị phân (.harenn) để tiếp tục train")
    parser.add_argument("--resume-epoch", type=int, default=None, help="Chỉ định epoch bắt đầu (mặc định: tự động nhận diện từ checkpoint + 1)")
    parser.add_argument("--reset-optimizer", action="store_true", help="Khởi tạo lại optimizer & LR schedule, chỉ dùng weights của model cũ")
    parser.add_argument("--skip-stage1", action="store_true", help="Bỏ qua Giai đoạn 1 và chuyển thẳng sang Stage 2 Fine-tune")
    parser.add_argument("--finetune-tactical", action="store_true", help="Chỉ finetune 3 đầu chiến thuật (tau, rho, rs), đóng băng toàn bộ backbone và eval_head")
    parser.add_argument("--tactical-epochs", type=int, default=2, help="Số epochs finetune đầu chiến thuật (mặc định: 2)")
    parser.add_argument("--output", type=str, default="Neuron Network/nextfish_hkv3.harenn", help="Đường dẫn xuất file .harenn")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print("=" * 76)
    print("  HALFKPV3 (HKV3) + HARENN TRAINING PIPELINE (SINGLE BIG NET ~88.85 MB)")
    print("=" * 76)
    if num_gpus > 0:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"  Thiết bị GPU:            {num_gpus}x GPU ({', '.join(gpu_names)}) | AMP FP16: {'ON' if use_amp else 'OFF'}")
    else:
        print("  Thiết bị:                CPU")
    print(f"  Không gian đặc trưng:    {HKV3_NUM_FEATURES:,} (32 Mirrored King Buckets x 11 Channels x 64)")
    print(f"  Virtual Factorizer:      W_full({HKV3_NUM_FEATURES:,}) + W_piece({HKV3_NUM_PIECE_FEATURES}) + W_rel({HKV3_NUM_REL_FEATURES:,})")
    print(f"  Cấu trúc tầng ẩn:        {HKV3_L1_SIZE} -> PairwiseGating({HKV3_L1_PAIR_SIZE}) -> SiameseTwinFC2({HKV3_L2_SIZE}) -> 4 Heads")
    print(f"  Đối xứng dữ liệu:        {args.symmetry}x Data Loader + Bit-Exact Algebraic Parity")
    print(f"  W_full Dropout:          {args.feature_dropout:.2%} (Chỉ áp dụng lên emb_full ở Stage 1, bảo toàn vật chất)")
    print(f"  Batch Size:              {args.batch_size:,} | LR: {args.lr:.2e} | Epochs: {args.epochs} + {args.finetune_eval_epochs} FT")
    if args.resume:
        print(f"  Resume Model:            {args.resume}")
    print(f"  File xuất nhị phân:      {args.output}")
    print("=" * 76, flush=True)

    t_train_start = time.time()
    telemetry = HFTelemetry(args.hf_log_repo, args.live_log_file)
    telemetry.push({"stage": "startup", "elapsed_s": 0.0})

    # 1. Đồng bộ dữ liệu từ Hugging Face Bucket nếu có cờ --sync-hf
    if args.sync_hf:
        sync_from_huggingface(args.hf_bucket, args.data)

    # 2. Khởi tạo bộ nạp dữ liệu: Streaming Multi-NPZ hoặc Single-NPZ truyền thống
    device_tmp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    streamer: Optional[MultiNPZStreamer] = None
    single_train_loader: Optional[FastHKV3BatchLoader] = None

    local_npz_files: List[Path] = []
    if args.cache_dir:
        cdir = Path(args.cache_dir)
        if not cdir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục --cache-dir: {cdir}")
        local_npz_files = sorted(
            p for p in cdir.rglob("*.npz") if not p.name.startswith("_hkv3_sparse_cache")
        )
    elif not args.cache and not args.kaggle_index and Path(args.data).is_dir():
        found_npz = sorted(
            p for p in Path(args.data).rglob("*.npz") if not p.name.startswith("_hkv3_sparse_cache")
        )
        if len(found_npz) >= 2:
            local_npz_files = found_npz

    if args.kaggle_index:
        streamer = MultiNPZStreamer(
            kaggle_index_ref=args.kaggle_index,
            batch_size=args.batch_size,
            symmetry=args.symmetry,
            chunk_max_pos=args.chunk_max_pos,
            start_part=args.start_part,
            end_part=args.end_part,
            device=device_tmp,
        )
        val_loader, val_us_sample, val_them_sample = streamer.get_val_loader()
        telemetry.push({
            "stage": "data_ready",
            "elapsed_s": round(time.time() - t_train_start, 1),
            "num_chunks": streamer.num_chunks(),
        })
    elif len(local_npz_files) >= 2:
        streamer = MultiNPZStreamer(
            local_npz_files=local_npz_files,
            batch_size=args.batch_size,
            symmetry=args.symmetry,
            chunk_max_pos=args.chunk_max_pos,
            device=device_tmp,
        )
        val_loader, val_us_sample, val_them_sample = streamer.get_val_loader()
    else:
        cache_arg = args.cache if args.cache else (str(local_npz_files[0]) if len(local_npz_files) == 1 else None)
        us_data, them_data, eval_data, wdl_data, eval_mask_data, tau_data, rho_data, rs_data, is_val_mask = load_or_build_hkv3_cache(
            args.data, cache_path=cache_arg, use_cache=True, val_split=args.val_split
        )

        n_total = len(eval_data)
        if is_val_mask is not None and np.any(is_val_mask == 1) and np.any(is_val_mask == 0):
            val_indices_all = np.flatnonzero(is_val_mask == 1)
            train_indices = np.flatnonzero(is_val_mask == 0)
            if len(val_indices_all) > 100000:
                rng_val = np.random.default_rng(42)
                val_indices = rng_val.choice(val_indices_all, size=100000, replace=False)
            else:
                val_indices = val_indices_all
            print("[Split] Sử dụng game/file-isolated Val split (0% leakage cùng ván).", flush=True)
        else:
            val_count = min(100000, max(1000, int(n_total * args.val_split)))
            val_indices = np.arange(n_total - val_count, n_total, dtype=np.int64)
            train_indices = np.arange(0, n_total - val_count, dtype=np.int64)

        print(f"[Split] Train: {len(train_indices):,} base ({len(train_indices) * args.symmetry:,} effective) | "
              f"Val: {len(val_indices):,} positions.", flush=True)

        single_train_loader = FastHKV3BatchLoader(
            us_data, them_data, eval_data, wdl_data, eval_mask_data, tau_data, rho_data, rs_data,
            train_indices, batch_size=args.batch_size, shuffle=True,
            symmetry=args.symmetry, device=device
        )
        val_loader = FastHKV3BatchLoader(
            us_data, them_data, eval_data, wdl_data, eval_mask_data, tau_data, rho_data, rs_data,
            val_indices, batch_size=args.batch_size, shuffle=False,
            symmetry=1, device=device
        )
        sample_n = min(64, len(val_indices))
        val_us_sample = us_data[val_indices[:sample_n]].copy()
        val_them_sample = them_data[val_indices[:sample_n]].copy()

    def iter_train_loaders(epoch_seed: int, start_chunk: int = 0):
        if streamer is not None:
            yield from streamer.iter_chunks(epoch_seed=epoch_seed, start_chunk=start_chunk)
        else:
            yield 0, 1, single_train_loader

    # 3. Khởi tạo mạng HalfKPv3 với Tensor-Parallel tự nhiên trên 2 GPU (ví dụ Kaggle T4x2)
    dev0 = torch.device("cuda:0") if num_gpus >= 1 else torch.device("cpu")
    dev1 = torch.device("cuda:1") if num_gpus >= 2 else dev0
    if num_gpus >= 2:
        print(f"[T4x2 Tensor-Parallel] Nhánh Left (1024) -> {dev0} || Nhánh Right (1024) -> {dev1} (Zero PCIe Weight Copy!)", flush=True)

    base_model = HalfKPv3Network(
        qat=not args.no_qat,
        dev0=dev0,
        dev1=dev1,
        full_dropout=args.feature_dropout,
        use_amp=use_amp,
    )

    model: nn.Module = base_model
    criterion = HKV3MultiTaskLoss(wdl_weight=args.wdl_weight, wdl_lambda=args.wdl_lambda).to(dev0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def active_emb_full_l2(us_b: torch.Tensor, them_b: torch.Tensor) -> torch.Tensor:
        uniq0 = torch.unique(torch.cat([us_b.reshape(-1), them_b.reshape(-1)]))
        uniq0 = uniq0[uniq0 < PADDING_FEATURE_IDX]
        if uniq0.numel() == 0:
            return torch.zeros((), device=dev0, dtype=torch.float32)
        l2_l = base_model.left_branch.emb_full.weight[uniq0].float().square().mean()
        uniq1 = uniq0.to(dev1) if dev1 != dev0 else uniq0
        l2_r = base_model.right_branch.emb_full.weight[uniq1].float().square().mean().to(dev0)
        return 0.5 * (l2_l + l2_r)

    # Phân nhóm tham số tối ưu:
    # - emb_full / psqt_full dùng weight_decay=0.0 trong AdamW để tránh bào mòn các ô Vua hiếm vắng mặt trong batch;
    #   thay bằng active_emb_full_l2() chỉ chuẩn hóa các đặc trưng thực sự xuất hiện trong batch.
    opt_groups = [
        {"params": list(base_model.left_branch.emb_full.parameters()) + list(base_model.right_branch.emb_full.parameters()), "lr": args.lr * 0.8, "weight_decay": 0.0},
        {"params": list(base_model.left_branch.emb_piece.parameters()) + list(base_model.right_branch.emb_piece.parameters()), "lr": args.lr, "weight_decay": 1e-5},
        {"params": list(base_model.left_branch.emb_rel.parameters()) + list(base_model.right_branch.emb_rel.parameters()), "lr": args.lr, "weight_decay": 1e-5},
        {"params": [base_model.left_branch.bias, base_model.right_branch.bias], "lr": args.lr, "weight_decay": 0.0},
        {"params": base_model.psqt_full.parameters(), "lr": args.lr * 0.5, "weight_decay": 0.0},
        {"params": base_model.psqt_piece.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.psqt_rel.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.fc2_self.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.fc2_opp.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.eval_head.parameters(), "lr": args.lr * 1.2, "weight_decay": 1e-6},
        {"params": base_model.tac_head.parameters(), "lr": args.lr * 1.2, "weight_decay": 1e-6},
    ]
    try:
        optimizer = optim.AdamW(opt_groups, fused=(dev0.type == "cuda"))
    except Exception:
        optimizer = optim.AdamW(opt_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=2e-5)

    start_epoch = 1
    resume_chunk_idx = 0
    best_val_loss = float("inf")
    best_val_eval = float("inf")
    best_val_mae = float("inf")
    best_pt_path = Path(args.output).with_suffix(".best.pt")

    # 4. Nạp checkpoint / mô hình cũ nếu có tham số --resume
    if args.resume:
        ckpt_dict, loaded_epoch = load_hkv3_weights_or_checkpoint(base_model, args.resume, dev0, dev1)
        in_prog_ep = ckpt_dict.get("in_progress_epoch")
        if args.resume_epoch is not None:
            start_epoch = args.resume_epoch
            print(f"[Resume] Ghi đè epoch bắt đầu: {start_epoch}", flush=True)
        elif in_prog_ep is not None and int(in_prog_ep) > 0:
            start_epoch = int(in_prog_ep)
            resume_chunk_idx = int(ckpt_dict.get("next_chunk_idx", 0))
            print(f"[Resume] Tiếp tục giữa Epoch {start_epoch} tại Chunk {resume_chunk_idx + 1}!", flush=True)
        elif loaded_epoch > 0:
            start_epoch = loaded_epoch + 1
            print(f"[Resume] Tự động tiếp tục từ epoch: {start_epoch}", flush=True)
        elif ckpt_dict.get("type") == "harenn":
            print(
                "[Resume Cảnh Báo] File .harenn không lưu số epoch trong header và tên file không có số epoch. "
                "Mặc định start_epoch = 1 (sẽ train từ đầu). "
                "Nếu muốn tiếp tục từ epoch cụ thể, hãy thêm tham số: --resume-epoch <N> hoặc --skip-stage1.",
                flush=True,
            )

        if not args.reset_optimizer:
            if "optimizer_state_dict" in ckpt_dict:
                try:
                    optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
                    print("[Resume] Đã khôi phục trạng thái AdamW optimizer thành công!", flush=True)
                except Exception as e:
                    print(f"[Resume Warning] Không thể khôi phục optimizer: {e}", flush=True)
            if "scaler_state_dict" in ckpt_dict and use_amp:
                try:
                    scaler.load_state_dict(ckpt_dict["scaler_state_dict"])
                    print("[Resume] Đã khôi phục trạng thái AMP GradScaler thành công!", flush=True)
                except Exception as e:
                    print(f"[Resume Warning] Không thể khôi phục scaler: {e}", flush=True)
            if "scheduler_state_dict" in ckpt_dict:
                try:
                    scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])
                    print("[Resume] Đã khôi phục trạng thái CosineAnnealingLR scheduler thành công!", flush=True)
                except Exception as e:
                    print(f"[Resume Warning] Không thể khôi phục scheduler: {e}", flush=True)
            elif start_epoch > 1:
                # Đồng bộ scheduler đến epoch hiện tại
                for _ in range(min(start_epoch - 1, args.epochs)):
                    scheduler.step()
                print(f"[Resume] Đã đồng bộ scheduler tới epoch {start_epoch} (LR: {scheduler.get_last_lr()[0]:.2e})", flush=True)

            if "best_val_loss" in ckpt_dict:
                best_val_loss = float(ckpt_dict["best_val_loss"])
            if "best_val_mae" in ckpt_dict:
                best_val_mae = float(ckpt_dict["best_val_mae"])
                print(f"[Resume] Điểm Val trước đó: Loss={best_val_loss:.4f}, MAE={best_val_mae:.2f} cp", flush=True)
        else:
            print("[Resume] Cờ --reset-optimizer được bật: Khởi tạo lại optimizer và LR schedule mới.", flush=True)

    def evaluate_validation() -> Dict[str, float]:
        model.eval()
        totals = {"total": 0.0, "eval": 0.0, "tau": 0.0, "rho": 0.0, "rs": 0.0, "mae_cp": 0.0, "wdl_err": 0.0}
        steps = 0
        with torch.no_grad():
            for us_b, them_b, targets_b in val_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs_b = model(us_b, them_b)
                    _, m = criterion(outputs_b, targets_b)
                for k in totals:
                    totals[k] += m[k]
                steps += 1
        for k in totals:
            totals[k] /= max(1, steps)
        return totals

    # ─── GIAI ĐOẠN ĐẶC BIỆT: CHUYÊN SÂU FINETUNE ĐẦU CHIẾN THUẬT (TAU, RHO, RS) ───
    if args.finetune_tactical:
        print("\n" + "=" * 76)
        print("  [Stage Tactical] CHUYÊN SÂU FINETUNE ĐẦU CHIẾN THUẬT (TAU, RHO, RS)")
        print("  Đóng băng toàn bộ Feature Extractor Backbone và Eval Head!")
        print("  Chỉ huấn luyện 3 Linear Heads: tau, rho, rs với lr=2.0e-3")
        print("=" * 76, flush=True)

        base_model.set_full_dropout(0.0)
        for p in base_model.parameters():
            p.requires_grad = False
        for p in base_model.tac_head.parameters():
            p.requires_grad = True

        tac_opt = optim.AdamW(base_model.tac_head.parameters(), lr=2.0e-3, weight_decay=1e-4)
        tac_sched = optim.lr_scheduler.CosineAnnealingLR(tac_opt, T_max=args.tactical_epochs, eta_min=1e-4)

        for ep in range(1, args.tactical_epochs + 1):
            model.train()
            t0 = time.time()
            tac_loss_sum = 0.0
            steps = 0
            for _, _, train_loader in iter_train_loaders(epoch_seed=1000 + ep):
                for us_b, them_b, targets_b in train_loader:
                    tac_opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs_b = model(us_b, them_b)
                        loss_tau = F.binary_cross_entropy_with_logits(outputs_b["tau"], targets_b["tau"])
                        loss_rho = F.binary_cross_entropy_with_logits(outputs_b["rho"], targets_b["rho"])
                        loss_rs = F.binary_cross_entropy_with_logits(outputs_b["rs"], targets_b["rs"])
                        loss = loss_tau + loss_rho + loss_rs

                    scaler.scale(loss).backward()
                    scaler.unscale_(tac_opt)
                    torch.nn.utils.clip_grad_norm_(base_model.tac_head.parameters(), max_norm=1.0)
                    scaler.step(tac_opt)
                    scaler.update()
                    tac_loss_sum += loss.item()
                    steps += 1

            tac_sched.step()
            dt = time.time() - t0
            val_m = evaluate_validation()
            print(
                f"[Tactical Ep {ep:02d}/{args.tactical_epochs:02d} ({dt:5.1f}s)] "
                f"Train BCE Loss: {tac_loss_sum / max(1, steps):.4f} | "
                f"Val: [tau:{val_m['tau']:.4f} rho:{val_m['rho']:.4f} rs:{val_m['rs']:.4f}] | Val MAE: {val_m['mae_cp']:.2f}cp",
                flush=True,
            )

        print(f"\n[Export] Xuất nhị phân cập nhật Tactical Heads: {args.output}...", flush=True)
        export_hkv3_binary(base_model, args.output)
        for p in base_model.parameters():
            p.requires_grad = True

        print(f"[Done] Tactical heads finetune hoàn tất! -> {args.output}", flush=True)
        return

    # ─── GIAI ĐOẠN 1: HUẤN LUYỆN ĐA NHIỆM ĐỒNG THỜI (EVAL + TAU + RHO + RS) ───
    if args.skip_stage1:
        print("\n[Stage 1] Bỏ qua Giai đoạn 1 do cờ --skip-stage1 được kích hoạt.", flush=True)
    elif start_epoch > args.epochs:
        print(
            f"\n[Stage 1] Checkpoint đã hoàn thành {start_epoch - 1}/{args.epochs} epochs. "
            f"Bỏ qua Stage 1 để chuyển tiếp sang Stage 2 (hoặc chỉ định --epochs lớn hơn để train tiếp).",
            flush=True,
        )
    else:
        print(f"\n[Stage 1] Bắt đầu huấn luyện Đa nhiệm HalfKPv3 (Epoch {start_epoch:02d} -> {args.epochs:02d})...", flush=True)
        time_budget_hit = False
        latest_path = Path(args.output).with_suffix(".latest.pt")
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            train_totals = {"total": 0.0, "eval": 0.0, "tau": 0.0, "rho": 0.0, "rs": 0.0, "mae_cp": 0.0}
            steps = 0
            c_start = resume_chunk_idx if epoch == start_epoch else 0
            resume_chunk_idx = 0
            epoch_finished = True
            nan_cnt = 0

            for c_idx, n_chunks, train_loader in iter_train_loaders(epoch_seed=42 + epoch, start_chunk=c_start):
                for us_b, them_b, targets_b in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs_b = model(us_b, them_b)
                        loss, m = criterion(outputs_b, targets_b)
                        if (steps & 7) == 0:
                            loss = loss + 8e-4 * active_emb_full_l2(us_b, them_b)

                    if not math.isfinite(m["total"]):
                        nan_cnt += 1
                        if nan_cnt >= 5:
                            telemetry.push({"stage": "error", "error": "NaN/Inf loss", "epoch": epoch, "chunk": c_idx + 1, "steps": steps}, wait=True)
                            raise RuntimeError(f"Phát hiện loss NaN/Inf {nan_cnt} lần liên tiếp tại Epoch {epoch}, Chunk {c_idx + 1}!")
                        continue
                    nan_cnt = 0

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    base_model.clamp_quantizable_weights()

                    for k in train_totals:
                        train_totals[k] += m[k]
                    steps += 1

                if streamer is not None and (c_idx + 1) < n_chunks:
                    val_mid = evaluate_validation()
                    best_val_loss = min(best_val_loss, val_mid["total"])
                    is_mid_best = val_mid["eval"] < best_val_eval
                    if is_mid_best:
                        best_val_eval = val_mid["eval"]
                        best_val_mae = val_mid["mae_cp"]
                    mid_ckpt = {
                        "epoch": epoch - 1,
                        "in_progress_epoch": epoch,
                        "next_chunk_idx": c_idx + 1,
                        "model_state_dict": base_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_mae": best_val_mae,
                    }
                    torch.save(mid_ckpt, latest_path)
                    if is_mid_best:
                        torch.save(mid_ckpt, best_pt_path)
                        export_hkv3_binary(base_model, args.output)
                    print(
                        f"  [Ckpt Ep {epoch:02d} Chunk {c_idx + 1}/{n_chunks}] "
                        f"Val Loss: {val_mid['total']:.4f} | MAE: {val_mid['mae_cp']:.1f}cp",
                        flush=True,
                    )
                    telemetry.push(
                        {
                            "stage": "stage1_chunk",
                            "epoch": epoch,
                            "chunk": c_idx + 1,
                            "total_chunks": n_chunks,
                            "steps": steps,
                            "elapsed_s": round(time.time() - t_train_start, 1),
                            "val_loss": round(val_mid["total"], 5),
                            "val_eval_loss": round(val_mid["eval"], 5),
                            "val_mae_cp": round(val_mid["mae_cp"], 2),
                            "val_wdl_err": round(val_mid["wdl_err"], 6),
                            "tau": round(val_mid["tau"], 5),
                            "rho": round(val_mid["rho"], 5),
                            "rs": round(val_mid["rs"], 5),
                        },
                        extra_files=[Path(args.output)] if is_mid_best else None,
                    )
                    if args.time_budget > 0 and (time.time() - t_train_start) >= args.time_budget:
                        print(f"[Time Budget] Đạt giới hạn {args.time_budget}s sau Chunk {c_idx + 1}/{n_chunks}. Lưu checkpoint để resume phiên sau!", flush=True)
                        epoch_finished = False
                        time_budget_hit = True
                        train_loader.release()
                        gc.collect()
                        break
                train_loader.release()
                gc.collect()

            if not epoch_finished:
                break
            if steps == 0:
                telemetry.push({"stage": "error", "error": "Zero training steps in epoch", "epoch": epoch}, wait=True)
                raise RuntimeError(f"Epoch {epoch} kết thúc với 0 steps train!")

            scheduler.step()
            dt = time.time() - t0
            for k in train_totals:
                train_totals[k] /= max(1, steps)

            val_m = evaluate_validation()
            cur_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:5.1f}s) | "
                f"Train Loss: {train_totals['total']:.4f} (MAE: {train_totals['mae_cp']:.1f}cp) | "
                f"Val Loss: {val_m['total']:.4f} (MAE: {val_m['mae_cp']:.1f}cp, WDL^2.5: {val_m['wdl_err']:.5f}) | "
                f"[tau:{val_m['tau']:.4f} rho:{val_m['rho']:.4f} rs:{val_m['rs']:.4f}] | LR: {cur_lr:.2e}",
                flush=True,
            )

            # Cập nhật best_val_eval / best_val_mae TRƯỚC khi đóng gói ckpt_data để không bỏ lỡ best chess model vì nhiễu aux loss
            best_val_loss = min(best_val_loss, val_m["total"])
            is_new_best = val_m["eval"] < best_val_eval
            if is_new_best:
                best_val_eval = val_m["eval"]
                best_val_mae = val_m["mae_cp"]

            ckpt_data = {
                "epoch": epoch,
                "in_progress_epoch": None,
                "next_chunk_idx": 0,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
            }
            ckpt_path = Path(args.output).with_suffix(f".epoch{epoch}.pt")
            torch.save(ckpt_data, ckpt_path)
            torch.save(ckpt_data, latest_path)

            if is_new_best:
                torch.save(ckpt_data, best_pt_path)
                export_hkv3_binary(base_model, args.output)

            telemetry.push(
                {
                    "stage": "stage1_epoch_end",
                    "epoch": epoch,
                    "steps": steps,
                    "elapsed_s": round(time.time() - t_train_start, 1),
                    "train_loss": round(train_totals["total"], 5),
                    "val_loss": round(val_m["total"], 5),
                    "val_mae_cp": round(val_m["mae_cp"], 2),
                    "val_wdl_err": round(val_m["wdl_err"], 6),
                    "tau": round(val_m["tau"], 5),
                    "rho": round(val_m["rho"], 5),
                    "rs": round(val_m["rs"], 5),
                    "lr": cur_lr,
                },
                extra_files=[Path(args.output)] if is_new_best else None,
            )

            if args.time_budget > 0 and (time.time() - t_train_start) >= args.time_budget:
                print(f"[Time Budget] Đạt giới hạn {args.time_budget}s cuối Epoch {epoch}. Dừng sạch để resume phiên sau!", flush=True)
                time_budget_hit = True
                break

        if time_budget_hit:
            print(f"\n[Paused] Phiên tạm dừng theo --time-budget. Chạy lại kèm `--resume {latest_path}` để train tiếp!", flush=True)
            telemetry.push({"stage": "paused_time_budget", "elapsed_s": round(time.time() - t_train_start, 1)}, extra_files=[latest_path, Path(args.output)], wait=True)
            telemetry.close()
            return

    # ─── GIAI ĐOẠN 2: TINH CHỈNH CHUYÊN SÂU ĐẦU ĐỌC EVAL (2A: JOINT REFINEMENT + 2B: LOCKED HEAD) ───
    if args.finetune_eval_epochs > 0:
        phase_a = max(1, args.finetune_eval_epochs // 2)
        phase_b = max(1, args.finetune_eval_epochs - phase_a)
        print(f"\n[Stage 2] Tinh chỉnh chuyên sâu Eval ({phase_a} epochs Phase 2A + {phase_b} epochs Phase 2B)...", flush=True)

        # Tắt hoàn toàn W_full dropout ở Stage 2 để mạng hội tụ chính xác từng centipawn
        base_model.set_full_dropout(0.0)

        # Khôi phục Best Model từ Stage 1 trước khi bước vào Stage 2A (tránh bắt đầu từ epoch cuối bị overfit)
        if best_pt_path.exists():
            print(f"[Stage 2A] Nạp lại Best Checkpoint từ Stage 1 ({best_pt_path})...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, str(best_pt_path), dev0, dev1)
        elif Path(args.output).exists():
            print(f"[Stage 2A] Nạp lại Best Binary từ Stage 1 ({args.output})...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, args.output, dev0, dev1)

        # Nếu chưa có mốc best_val_mae (ví dụ khi chạy --skip-stage1 từ file .harenn), đo ngay trên tập Val
        if not math.isfinite(best_val_mae):
            init_val = evaluate_validation()
            best_val_loss = init_val["total"]
            best_val_mae = init_val["mae_cp"]
            print(f"[Stage 2A] Baseline Val MAE trước khi fine-tune: {best_val_mae:.2f} cp", flush=True)

        # Phase 2A: Ưu tiên Eval gấp 4 lần nhưng vẫn giữ neo tau/rho/rs
        ft_opt_a = optim.AdamW([
            {"params": list(base_model.left_branch.emb_full.parameters()) + list(base_model.right_branch.emb_full.parameters()), "lr": 1.0e-4, "weight_decay": 0.0},
            {"params": list(base_model.left_branch.emb_piece.parameters()) + list(base_model.right_branch.emb_piece.parameters()), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": list(base_model.left_branch.emb_rel.parameters()) + list(base_model.right_branch.emb_rel.parameters()), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": [base_model.left_branch.bias, base_model.right_branch.bias], "lr": 1.0e-4, "weight_decay": 0.0},
            {"params": base_model.psqt_full.parameters(), "lr": 1.0e-4, "weight_decay": 0.0},
            {"params": base_model.psqt_piece.parameters(), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": base_model.psqt_rel.parameters(), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": base_model.fc2_self.parameters(), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": base_model.fc2_opp.parameters(), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": base_model.eval_head.parameters(), "lr": 3.0e-4, "weight_decay": 1e-6},
            {"params": base_model.tac_head.parameters(), "lr": 5.0e-5, "weight_decay": 1e-6},
        ])
        ft_sched_a = optim.lr_scheduler.CosineAnnealingLR(ft_opt_a, T_max=phase_a, eta_min=1e-5)

        for ep in range(1, phase_a + 1):
            model.train()
            t0 = time.time()
            for _, _, train_loader in iter_train_loaders(epoch_seed=2000 + ep):
                for us_b, them_b, targets_b in train_loader:
                    ft_opt_a.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs_b = model(us_b, them_b)
                        loss_ev = criterion.compute_eval_loss(
                            outputs_b["eval"], targets_b["eval"],
                            target_wdl_game=targets_b.get("wdl"),
                            eval_mask=targets_b.get("eval_mask"),
                        )
                        loss_aux = (
                            F.binary_cross_entropy_with_logits(outputs_b["tau"], targets_b["tau"])
                            + F.binary_cross_entropy_with_logits(outputs_b["rho"], targets_b["rho"])
                            + F.binary_cross_entropy_with_logits(outputs_b["rs"], targets_b["rs"])
                        )
                        loss = 4.0 * loss_ev + 0.15 * loss_aux + 1e-4 * active_emb_full_l2(us_b, them_b)
                    scaler.scale(loss).backward()
                    scaler.unscale_(ft_opt_a)
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=0.5)
                    scaler.step(ft_opt_a)
                    scaler.update()
                    base_model.clamp_quantizable_weights()
                train_loader.release()
                gc.collect()

            ft_sched_a.step()
            dt = time.time() - t0
            val_m = evaluate_validation()
            print(
                f"[Phase 2A {ep:02d}/{phase_a:02d} ({dt:5.1f}s)] "
                f"Val Eval Loss: {val_m['eval']:.4f} | Val MAE: {val_m['mae_cp']:.2f}cp | WDL^2.5: {val_m['wdl_err']:.5f}",
                flush=True,
            )
            if val_m["mae_cp"] < best_val_mae:
                best_val_mae = val_m["mae_cp"]
                torch.save({
                    "stage": "2A_best",
                    "model_state_dict": base_model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_mae": best_val_mae,
                }, best_pt_path)
                export_hkv3_binary(base_model, args.output)

        ckpt_2a_path = Path(args.output).with_suffix(".phase2a.pt")
        torch.save({
            "stage": "2A",
            "model_state_dict": base_model.state_dict(),
            "best_val_mae": best_val_mae,
        }, ckpt_2a_path)

        # Khôi phục Best Model sau Phase 2A trước khi đóng băng Backbone sang Phase 2B
        if best_pt_path.exists():
            print(f"[Stage 2B] Nạp lại Best Checkpoint ({best_pt_path}, MAE: {best_val_mae:.2f} cp) trước khi khóa Backbone...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, str(best_pt_path), dev0, dev1)
        elif Path(args.output).exists():
            print(f"[Stage 2B] Nạp lại Best Binary ({args.output}, MAE: {best_val_mae:.2f} cp) trước khi khóa Backbone...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, args.output, dev0, dev1)

        # Phase 2B: Đóng băng toàn bộ Backbone & tac_head, chỉ hiệu chỉnh eval_head (0.00% drift trên tau/rho/rs)
        for p in base_model.parameters():
            p.requires_grad = False
        for p in base_model.eval_head.parameters():
            p.requires_grad = True

        ft_opt_b = optim.AdamW(base_model.eval_head.parameters(), lr=1.5e-4, weight_decay=0.0)
        ft_sched_b = optim.lr_scheduler.CosineAnnealingLR(ft_opt_b, T_max=phase_b, eta_min=5e-6)

        for ep in range(1, phase_b + 1):
            model.train()
            t0 = time.time()
            for _, _, train_loader in iter_train_loaders(epoch_seed=3000 + ep):
                for us_b, them_b, targets_b in train_loader:
                    ft_opt_b.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs_b = model(us_b, them_b)
                        loss_ev = criterion.compute_eval_loss(
                            outputs_b["eval"], targets_b["eval"],
                            target_wdl_game=targets_b.get("wdl"),
                            eval_mask=targets_b.get("eval_mask"),
                        )
                    scaler.scale(loss_ev).backward()
                    scaler.unscale_(ft_opt_b)
                    scaler.step(ft_opt_b)
                    scaler.update()
                    base_model.clamp_quantizable_weights()
                train_loader.release()
                gc.collect()

            ft_sched_b.step()
            dt = time.time() - t0
            val_m = evaluate_validation()
            print(
                f"[Phase 2B {ep:02d}/{phase_b:02d} ({dt:5.1f}s)] "
                f"Val Eval Loss: {val_m['eval']:.4f} | Val MAE: {val_m['mae_cp']:.2f}cp | WDL^2.5: {val_m['wdl_err']:.5f}",
                flush=True,
            )
            if val_m["mae_cp"] < best_val_mae:
                best_val_mae = val_m["mae_cp"]
                torch.save({
                    "stage": "2B_best",
                    "model_state_dict": base_model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_mae": best_val_mae,
                }, best_pt_path)
                export_hkv3_binary(base_model, args.output)

        ckpt_2b_path = Path(args.output).with_suffix(".phase2b.pt")
        torch.save({
            "stage": "2B",
            "model_state_dict": base_model.state_dict(),
            "best_val_mae": best_val_mae,
        }, ckpt_2b_path)

        for p in base_model.parameters():
            p.requires_grad = True

    # ─── KIỂM CHỨNG ĐỐI XỨNG SỐ NGUYÊN C++ AVX2 SAU KHI XUẤT FILE ───
    # Nạp lại chính xác Best Model đã lưu để tránh model ở epoch cuối ghi đè làm mất best model!
    if best_pt_path.exists():
        print(f"\n[Verification] Nạp lại Best Model (MAE: {best_val_mae:.2f} cp) từ {best_pt_path} để kiểm chứng...", flush=True)
        _, _ = load_hkv3_weights_or_checkpoint(base_model, str(best_pt_path), dev0, dev1)
    elif Path(args.output).exists():
        print(f"\n[Verification] Nạp lại Best Model (MAE: {best_val_mae:.2f} cp) từ {args.output} để kiểm chứng...", flush=True)
        _, _ = load_hkv3_weights_or_checkpoint(base_model, args.output, dev0, dev1)
    quant_tensors = export_hkv3_binary(base_model, args.output)
    sample_n = min(64, len(val_us_sample))
    verify_integer_avx2_parity_and_accuracy(
        quant_tensors,
        val_us_sample[:sample_n],
        val_them_sample[:sample_n],
        model=base_model,
    )
    print(f"\n[Done] Huấn luyện hoàn tất! Best Validation MAE: {best_val_mae:.2f} cp -> {args.output}", flush=True)
    telemetry.push({"stage": "done", "best_val_mae_cp": round(best_val_mae, 2), "elapsed_s": round(time.time() - t_train_start, 1)}, extra_files=[Path(args.output)], wait=True)
    telemetry.close()


if __name__ == "__main__":
    main()
