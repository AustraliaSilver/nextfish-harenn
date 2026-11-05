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
import contextlib
import gc
import glob
import hashlib
import json
import math
import os
import queue
import random
import re
import struct
import sys
import threading
import time
import warnings
import zlib
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ẩn cảnh báo không cần thiết của PyTorch autograd khi dùng Straight-Through Estimator (QAT) trên TPU
warnings.filterwarnings("ignore", message=".*The .grad attribute of a Tensor that is not a leaf Tensor.*")

try:
    import torch_tpu  # noqa: F401
except Exception:
    pass

HAS_NATIVE_TPU = hasattr(torch, "tpu")

HAS_XLA = False
if not HAS_NATIVE_TPU:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        try:
            import torch_xla.runtime as xr
        except Exception:
            xr = None
        HAS_XLA = True
    except Exception:
        HAS_XLA = False

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
WDL_SCALE_PAWNS = 4.0                   # 4.00 pawns = 400 centipawns for WDL sigmoid

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


def _parse_single_data_file_worker(file_path_str: str):
    """Hàm worker chạy đa tiến trình để phân tích 1 file .jsonl hoặc .json sang chỉ số HKV3."""
    path = Path(file_path_str)
    us_list = []
    them_list = []
    eval_list = []
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
        board_key = " ".join(parts[:3]) if len(parts) >= 3 else fen
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
            ev_cp = float(ev_raw)
        except (ValueError, TypeError):
            return

        ev_cp = max(-1000.0, min(1000.0, ev_cp))
        ev_pawns = ev_cp / 100.0

        tau = max(0.0, min(1.0, float(pos.get("tau", 0.5))))
        rho = max(0.0, min(1.0, float(pos.get("rho", 0.3))))
        rs = max(0.0, min(1.0, float(pos.get("rs", 0.5))))

        hash_list.append(h)
        us_list.append(us_arr)
        them_list.append(them_arr)
        eval_list.append(ev_pawns)
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

    return (
        np.asarray(hash_list, dtype=np.int64),
        np.vstack(us_list).astype(np.uint16, copy=False),
        np.vstack(them_list).astype(np.uint16, copy=False),
        np.asarray(eval_list, dtype=np.float32),
        np.asarray(tau_list, dtype=np.float32),
        np.asarray(rho_list, dtype=np.float32),
        np.asarray(rs_list, dtype=np.float32),
    )


def load_or_build_hkv3_cache(data_dir: str, cache_path: Optional[str] = None, use_cache: bool = True):
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
        default_cache = data_path / "_hkv3_sparse_cache_v1.npz"
        if default_cache.exists():
            cpath = default_cache
        else:
            npz_files = list(data_path.glob("*.npz"))
            cpath = npz_files[0] if npz_files else default_cache
    else:
        cpath = data_path / "_hkv3_sparse_cache_v1.npz"

    if cpath.exists() and use_cache:
        t0 = time.time()
        print(f"[Cache] Đang nạp cache nhị phân HKV3 từ {cpath}...", flush=True)
        try:
            gc.collect()
            with np.load(str(cpath), allow_pickle=False) as data:
                us_indices = data["us"] if "us" in data else data["us_indices"]
                them_indices = data["them"] if "them" in data else data["them_indices"]
                labels_eval = data["eval"] if "eval" in data else data["labels_eval"]
                labels_tau = data["tau"] if "tau" in data else data["labels_tau"]
                labels_rho = data["rho"] if "rho" in data else data["labels_rho"]
                labels_rs = data["rs"] if "rs" in data else data["labels_rs"]
            gc.collect()

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
                # Lưu ý: Trong không gian 64-King cũ (0..45055), chỉ số padding là >= 45056 (đã được khởi tạo
                # bằng PADDING_FEATURE_IDX trong np.full ở trên). Tuyệt đối KHÔNG gán lut[22528] = 22528
                # vì 22528 là đặc trưng hợp lệ của 64-King (ksq=32 [a5], ch=0 [Tốt mình], sq=0 [a1])!
                us_indices = lut[us_indices]
                them_indices = lut[them_indices]
                print(f"[Cache Auto-Fix] Chuẩn hóa thành công trong {time.time() - t_fix:.2f}s! (Max index mới: {max(int(us_indices.max()), int(them_indices.max()))})", flush=True)

            print(f"[Cache] Đã nạp {len(labels_eval):,} thế cờ duy nhất trong {time.time() - t0:.2f}s!", flush=True)
            return us_indices, them_indices, labels_eval, labels_tau, labels_rho, labels_rs
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

    file_strs = [str(f) for f in source_files]
    with Pool(workers) as pool:
        results = pool.map(_parse_single_data_file_worker, file_strs)

    valid_results = [r for r in results if r is not None]
    del results
    if not valid_results:
        raise RuntimeError("Không trích xuất được thế cờ hợp lệ nào từ các file dữ liệu!")

    all_hashes = np.concatenate([r[0] for r in valid_results], axis=0)
    all_us = np.concatenate([r[1] for r in valid_results], axis=0)
    all_them = np.concatenate([r[2] for r in valid_results], axis=0)
    all_eval = np.concatenate([r[3] for r in valid_results], axis=0)
    all_tau = np.concatenate([r[4] for r in valid_results], axis=0)
    all_rho = np.concatenate([r[5] for r in valid_results], axis=0)
    all_rs = np.concatenate([r[6] for r in valid_results], axis=0)
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

    # Tính kỳ vọng trung bình (mean) của các nhãn khi 1 thế cờ xuất hiện nhiều lần trong các ván khác nhau:
    # Tránh việc thế cờ bị gán nhãn thiên lệch do chỉ lấy ngẫu nhiên bản ghi đầu tiên!
    counts_f = counts.astype(np.float32)
    del counts
    labels_eval = (np.bincount(inv_idx, weights=all_eval) / counts_f).astype(np.float32)
    del all_eval
    labels_tau = (np.bincount(inv_idx, weights=all_tau) / counts_f).astype(np.float32)
    del all_tau
    labels_rho = (np.bincount(inv_idx, weights=all_rho) / counts_f).astype(np.float32)
    del all_rho
    labels_rs = (np.bincount(inv_idx, weights=all_rs) / counts_f).astype(np.float32)
    del all_rs, inv_idx, counts_f
    gc.collect()

    dt = time.time() - t0
    print(f"[Dataset] Hoàn tất lọc trùng: giữ lại {len(labels_eval):,} thế cờ duy nhất "
          f"(từ {total_raw:,} bản ghi thô) trong {dt:.2f}s ({total_raw / max(1e-6, dt):,.0f} pos/s).", flush=True)

    if use_cache:
        try:
            t_save = time.time()
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                str(cpath),
                us_indices=us_indices,
                them_indices=them_indices,
                labels_eval=labels_eval,
                labels_tau=labels_tau,
                labels_rho=labels_rho,
                labels_rs=labels_rs,
            )
            sz_mb = cpath.stat().st_size / (1024 * 1024)
            print(f"[Cache] Đã lưu cache nhị phân tốc độ cao tại {cpath} ({sz_mb:.1f} MB, {time.time() - t_save:.1f}s).", flush=True)
        except Exception as e:
            print(f"[Cache] Cảnh báo không thể ghi file cache: {e}", flush=True)

    return us_indices, them_indices, labels_eval, labels_tau, labels_rho, labels_rs


# ═══════════════════════════════════════════════════════════════════════════
# 4. BỘ NẠP BATCH SIÊU TỐC TRÊN GPU VỚI PINNED MEMORY DMA
# ═══════════════════════════════════════════════════════════════════════════
class FastHKV3BatchLoader:
    """
    Bộ nạp dữ liệu trực tiếp từ bộ nhớ NumPy sang GPU / TPU v5e-1 với:
      - Pinned Memory DMA (bất đồng bộ thực sự qua PCIe với non_blocking=True trên CUDA).
      - Truyền uint16 (2 bytes) qua PCIe rồi ép kiểu int64 ngay trên GPU/TPU HBM (giảm 75% băng thông PCIe).
      - Strictly Static Batch Shape trên TPU/XLA (tránh 100% lỗi biên dịch lại đồ thị HLO ở batch cuối).
      - Background Thread Prefetching giúp chồng lấp hoàn toàn việc cắt mảng NumPy trên CPU với tính toán MXU/GPU.
      - Mặc định symmetry=1 vì kiến trúc HalfKPv3 đã thỏa mãn đối xứng đại số tuyệt đối (E(them, us) == -E(us, them)).
    """

    def __init__(
        self,
        us_data: np.ndarray,
        them_data: np.ndarray,
        eval_data: np.ndarray,
        tau_data: np.ndarray,
        rho_data: np.ndarray,
        rs_data: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        symmetry: int = 1,
        device: torch.device = torch.device("cuda"),
        static_shape: Optional[bool] = None,
    ):
        self.us_data = np.asarray(us_data)
        self.them_data = np.asarray(them_data)
        self.eval_data = np.asarray(eval_data)
        self.tau_data = np.asarray(tau_data)
        self.rho_data = np.asarray(rho_data)
        self.rs_data = np.asarray(rs_data)
        self.base_indices = np.asarray(indices, dtype=np.int64)
        self.num_base = len(self.base_indices)
        self.symmetry = symmetry if symmetry in (1, 2) else 1
        self.total_samples = self.num_base * self.symmetry
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.is_xla = (self.device.type in ("tpu", "xla"))
        self.pin_memory = (self.device.type == "cuda")
        self.static_shape = self.is_xla if static_shape is None else static_shape

    def __len__(self) -> int:
        return max(1, (self.total_samples + self.batch_size - 1) // self.batch_size)

    def _to_device(self, arr: np.ndarray, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if self.is_xla and target_dtype == torch.int64:
            # Trên TPU/XLA, ép sang int64 từ mảng con kích thước 1 batch trên CPU hoặc truyền trực tiếp vào XLA tensor
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
            return t.to(self.device)
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if self.pin_memory:
            t = t.pin_memory()
        t_dev = t.to(self.device, non_blocking=not self.is_xla)
        if target_dtype is not None and t_dev.dtype != target_dtype:
            t_dev = t_dev.to(target_dtype)
        return t_dev

    def _prepare_cpu_batch(self, v_ids: np.ndarray):
        if self.symmetry == 2:
            is_flipped = (v_ids >= self.num_base)
            raw_pos = v_ids % self.num_base
            actual_ids = self.base_indices[raw_pos]
            us_np = self.us_data[actual_ids]
            them_np = self.them_data[actual_ids]
            ev_np = self.eval_data[actual_ids]
            if np.any(is_flipped):
                us_np = np.where(is_flipped[:, None], self.them_data[actual_ids], us_np)
                them_np = np.where(is_flipped[:, None], self.us_data[actual_ids], them_np)
                ev_np = np.where(is_flipped, -ev_np, ev_np)
        else:
            actual_ids = self.base_indices[v_ids]
            us_np = self.us_data[actual_ids]
            them_np = self.them_data[actual_ids]
            ev_np = self.eval_data[actual_ids]

        tau_np = self.tau_data[actual_ids]
        rho_np = self.rho_data[actual_ids]
        rs_np = self.rs_data[actual_ids]
        return us_np, them_np, ev_np, tau_np, rho_np, rs_np

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.total_samples)
        else:
            perm = np.arange(self.total_samples, dtype=np.int64)

        # Trên TPU v5e-1 (XLA), bắt buộc mọi batch (kể cả batch cuối cùng) có kích thước cố định [batch_size, 32]
        # để tránh XLA phải biên dịch lại đồ thị HLO (tốn 30-60s).
        if self.static_shape and len(perm) > 0:
            rem = len(perm) % self.batch_size
            if rem != 0:
                pad_needed = self.batch_size - rem
                pad_ids = perm[np.arange(pad_needed) % len(perm)]
                perm = np.concatenate([perm, pad_ids])

        batch_q: "queue.Queue" = queue.Queue(maxsize=3)
        sentinel = object()

        def _producer():
            try:
                for start in range(0, len(perm), self.batch_size):
                    v_ids = perm[start:start + self.batch_size]
                    batch_q.put(self._prepare_cpu_batch(v_ids))
            except Exception as exc:
                batch_q.put(exc)
            finally:
                batch_q.put(sentinel)

        prod_thread = threading.Thread(target=_producer, daemon=True)
        prod_thread.start()

        while True:
            item = batch_q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            us_np, them_np, ev_np, tau_np, rho_np, rs_np = item
            us_t = self._to_device(us_np, target_dtype=torch.int64)
            them_t = self._to_device(them_np, target_dtype=torch.int64)
            targets = {
                "eval": self._to_device(ev_np),
                "tau": self._to_device(tau_np),
                "rho": self._to_device(rho_np),
                "rs": self._to_device(rs_np),
            }
            yield us_t, them_t, targets


# ═══════════════════════════════════════════════════════════════════════════
# 5. KIẾN TRÚC MẠNG NƠ-RON ĐA NHIỆM HALFKPV3 (HKV3) CHUẨN ĐẠI SỐ & SIMD
# ═══════════════════════════════════════════════════════════════════════════
def _bag_sum(emb: nn.EmbeddingBag, idx: torch.Tensor) -> torch.Tensor:
    """
    Thực thi phép cộng nhúng thưa (EmbeddingBag mode='sum'):
      - Trên TPU (native 'tpu' hoặc 'xla'): Hạ trực tiếp xuống F.embedding(...).sum(dim=1) (native Gather + ReduceSum trên HBM2e 819 GB/s).
      - Trên CUDA/CPU: Gọi kernel C++/CUDA nn.EmbeddingBag nguyên bản.
    """
    if idx.device.type in ("tpu", "xla"):
        return F.embedding(idx, emb.weight, padding_idx=emb.padding_idx).sum(dim=1)
    return emb(idx)


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

    def forward(
        self,
        us_idx: torch.Tensor,
        them_idx: torch.Tensor,
        shared_maps: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if shared_maps is not None:
            us_p, us_r, them_p, them_r = shared_maps
        else:
            us_p = self.piece_map_pad[us_idx]
            us_r = self.rel_map_pad[us_idx]
            them_p = self.piece_map_pad[them_idx]
            them_r = self.rel_map_pad[them_idx]

        # Chỉ dropout trên bảng dư emb_full để ép mạng học quy luật tổng quát qua emb_piece & emb_rel
        # mà không làm mất quân cờ hay làm lệch vật chất giữa us và them!
        if self.training and self.full_dropout > 0.0:
            pad_val = us_idx.new_full((), PADDING_FEATURE_IDX)
            drop_u = (torch.rand_like(us_idx, dtype=torch.float32) < self.full_dropout) & (us_idx < PADDING_FEATURE_IDX)
            drop_v = (torch.rand_like(them_idx, dtype=torch.float32) < self.full_dropout) & (them_idx < PADDING_FEATURE_IDX)
            us_idx_full = torch.where(drop_u, pad_val, us_idx)
            them_idx_full = torch.where(drop_v, pad_val, them_idx)
        else:
            us_idx_full = us_idx
            them_idx_full = them_idx

        acc_us = _bag_sum(self.emb_full, us_idx_full) + _bag_sum(self.emb_piece, us_p) + _bag_sum(self.emb_rel, us_r) + self.bias
        acc_them = _bag_sum(self.emb_full, them_idx_full) + _bag_sum(self.emb_piece, them_p) + _bag_sum(self.emb_rel, them_r) + self.bias

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
      - Hỗ trợ tối ưu đồ thị tĩnh XLA + bfloat16 trên Google Cloud / Colab TPU v5e-1
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

    def _compute_psqt(
        self,
        idx: torch.Tensor,
        p_idx: Optional[torch.Tensor] = None,
        r_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if p_idx is None:
            p_idx = self.piece_map_pad[idx]
        if r_idx is None:
            r_idx = self.rel_map_pad[idx]
        return _bag_sum(self.psqt_full, idx) + _bag_sum(self.psqt_piece, p_idx) + _bag_sum(self.psqt_rel, r_idx)

    def forward(self, us_idx: torch.Tensor, them_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Bước 1: Chạy song song nhánh Left trên dev0 và nhánh Right trên dev1 (Tensor-Parallel trên T4x2)
        # hoặc dùng chung bảng tra chỉ số (Shared Factorizer Index Gather) khi chạy trên 1 chip như TPU v5e-1.
        us_p = self.piece_map_pad[us_idx]
        us_r = self.rel_map_pad[us_idx]
        them_p = self.piece_map_pad[them_idx]
        them_r = self.rel_map_pad[them_idx]
        shared_maps_0 = (us_p, us_r, them_p, them_r)

        if self.dev1 != self.dev0:
            is_dev1_tpu = (self.dev1.type in ("tpu", "xla"))
            amp_enabled = (us_idx.is_cuda or is_dev1_tpu) and self.use_amp
            us_idx_1 = us_idx.to(self.dev1, non_blocking=not is_dev1_tpu)
            them_idx_1 = them_idx.to(self.dev1, non_blocking=not is_dev1_tpu)
            if is_dev1_tpu:
                u_right, v_right = self.right_branch(us_idx_1, them_idx_1)
            else:
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    u_right, v_right = self.right_branch(us_idx_1, them_idx_1)
            u_left, v_left = self.left_branch(us_idx, them_idx, shared_maps=shared_maps_0)
            # Nén sang BF16 (trên TPU ICI) hoặc FP16 (trên GPU PCIe) trước khi truyền từ dev1 về dev0:
            if amp_enabled:
                xfer_dtype = torch.bfloat16 if is_dev1_tpu else torch.float16
                u_right = u_right.to(dtype=xfer_dtype).to(self.dev0, non_blocking=not is_dev1_tpu).to(dtype=u_left.dtype)
                v_right = v_right.to(dtype=xfer_dtype).to(self.dev0, non_blocking=not is_dev1_tpu).to(dtype=v_left.dtype)
            else:
                u_right = u_right.to(self.dev0, non_blocking=not is_dev1_tpu)
                v_right = v_right.to(self.dev0, non_blocking=not is_dev1_tpu)
        else:
            u_left, v_left = self.left_branch(us_idx, them_idx, shared_maps=shared_maps_0)
            u_right, v_right = self.right_branch(us_idx, them_idx, shared_maps=shared_maps_0)

        # Bước 2: Pairwise Bilinear Gating khớp chính xác phép dịch >> 7 (chia 128 floor) của C++ AVX2
        u = bilinear_gate_ste(u_left, u_right, self.qat)
        v = bilinear_gate_ste(v_left, v_right, self.qat)

        psqt_us = self._compute_psqt(us_idx, us_p, us_r)
        psqt_them = self._compute_psqt(them_idx, them_p, them_r)

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
      - Giữ toàn bộ metrics dưới dạng Tensor trên GPU/TPU trong suốt epoch (Zero Host-Device Sync .item() mỗi step!),
        giúp TPU v5e-1 chạy bất đồng bộ liên tục với hiệu suất tối đa.
    """

    def __init__(self, wdl_weight: float = 0.75, wdl_scale_pawns: float = WDL_SCALE_PAWNS):
        super().__init__()
        self.wdl_weight = wdl_weight
        self.wdl_scale_pawns = wdl_scale_pawns

    def compute_eval_loss(self, pred_pawns: torch.Tensor, target_pawns: torch.Tensor) -> torch.Tensor:
        target_clamped = torch.clamp(target_pawns, -10.0, 10.0)

        # 1. Smooth-L1 Huber Loss trực tiếp trên đơn vị Pawns (beta = 0.5 pawns = 50 cp)
        # Viết bằng các toán tử cơ bản (abs, where, square) để chạy native 100% trên cả TPU (torch_tpu/XLA) và GPU CUDA:
        #   Khi |diff| < 0.5: 0.5 * diff^2 / 0.5 == diff^2
        #   Khi |diff| >= 0.5: |diff| - 0.5 * 0.5 == |diff| - 0.25
        diff = torch.abs(pred_pawns - target_clamped)
        loss_huber = torch.where(diff < 0.5, diff.square(), diff - 0.25).mean()

        # 2. WDL Sigmoid Binary Cross-Entropy (with logits để an toàn tuyệt đối với FP16/BF16 autocast)
        target_wdl = torch.sigmoid(target_clamped / self.wdl_scale_pawns)
        loss_wdl = F.binary_cross_entropy_with_logits(
            pred_pawns / self.wdl_scale_pawns, target_wdl, reduction="mean"
        )

        return self.wdl_weight * loss_wdl + (1.0 - self.wdl_weight) * loss_huber

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_eval = self.compute_eval_loss(outputs["eval"], targets["eval"])

        loss_tau = F.binary_cross_entropy_with_logits(outputs["tau"], targets["tau"], reduction="mean")
        loss_rho = F.binary_cross_entropy_with_logits(outputs["rho"], targets["rho"], reduction="mean")
        loss_rs = F.binary_cross_entropy_with_logits(outputs["rs"], targets["rs"], reduction="mean")

        total_loss = loss_eval + 0.15 * (loss_tau + loss_rho + loss_rs)

        with torch.no_grad():
            eval_out = outputs["eval"].float()
            eval_tgt = targets["eval"].float()
            mae_cp = torch.mean(torch.abs(eval_out - eval_tgt)) * 100.0
            pred_wdl = torch.sigmoid(eval_out / self.wdl_scale_pawns)
            tgt_wdl = torch.sigmoid(eval_tgt / self.wdl_scale_pawns)
            wdl_err = torch.mean(torch.abs(pred_wdl - tgt_wdl) ** 2.5)

        metrics = {
            "total": total_loss.detach().float(),
            "eval": loss_eval.detach().float(),
            "tau": loss_tau.detach().float(),
            "rho": loss_rho.detach().float(),
            "rs": loss_rs.detach().float(),
            "mae_cp": mae_cp.detach(),
            "wdl_err": wdl_err.detach(),
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
            # Phân rã ngược L1 về 3 tầng (W_full, W_piece, W_rel) trên CPU (tránh biên dịch XLA one-off) rồi nạp lên dev0/dev1
            rel_map_cpu = model.left_branch.rel_map_pad[:HKV3_NUM_FEATURES].cpu()
            left_flat = torch.from_numpy(fc1_w[:, :HKV3_L1_PAIR_SIZE].astype(np.float32) / QA_SCALE)
            l_full, l_piece, l_rel = _decompose_triscale_table(left_flat, rel_map_cpu)
            model.left_branch.emb_full.weight.data[:HKV3_NUM_FEATURES] = l_full.to(dev0)
            model.left_branch.emb_full.weight.data[HKV3_NUM_FEATURES:].zero_()
            model.left_branch.emb_piece.weight.data[:HKV3_NUM_PIECE_FEATURES] = l_piece.to(dev0)
            model.left_branch.emb_piece.weight.data[HKV3_NUM_PIECE_FEATURES:].zero_()
            model.left_branch.emb_rel.weight.data[:HKV3_NUM_REL_FEATURES] = l_rel.to(dev0)
            model.left_branch.emb_rel.weight.data[HKV3_NUM_REL_FEATURES:].zero_()
            model.left_branch.bias.data = torch.from_numpy(fc1_b[:HKV3_L1_PAIR_SIZE].astype(np.float32) / QA_SCALE).to(dev0)

            right_flat = torch.from_numpy(fc1_w[:, HKV3_L1_PAIR_SIZE:].astype(np.float32) / QA_SCALE)
            r_full, r_piece, r_rel = _decompose_triscale_table(right_flat, rel_map_cpu)
            model.right_branch.emb_full.weight.data[:HKV3_NUM_FEATURES] = r_full.to(dev1)
            model.right_branch.emb_full.weight.data[HKV3_NUM_FEATURES:].zero_()
            model.right_branch.emb_piece.weight.data[:HKV3_NUM_PIECE_FEATURES] = r_piece.to(dev1)
            model.right_branch.emb_piece.weight.data[HKV3_NUM_PIECE_FEATURES:].zero_()
            model.right_branch.emb_rel.weight.data[:HKV3_NUM_REL_FEATURES] = r_rel.to(dev1)
            model.right_branch.emb_rel.weight.data[HKV3_NUM_REL_FEATURES:].zero_()
            model.right_branch.bias.data = torch.from_numpy(fc1_b[HKV3_L1_PAIR_SIZE:].astype(np.float32) / QA_SCALE).to(dev1)

            # Phân rã ngược PSQT về 3 tầng (psqt_full, psqt_piece, psqt_rel)
            psqt_flat = torch.from_numpy(psqt.astype(np.float32) / OUTPUT_SCALE)
            p_full, p_piece, p_rel = _decompose_triscale_table(psqt_flat, rel_map_cpu)
            model.psqt_full.weight.data[:HKV3_NUM_FEATURES] = p_full.to(dev0)
            model.psqt_full.weight.data[HKV3_NUM_FEATURES:].zero_()
            model.psqt_piece.weight.data[:HKV3_NUM_PIECE_FEATURES] = p_piece.to(dev0)
            model.psqt_piece.weight.data[HKV3_NUM_PIECE_FEATURES:].zero_()
            model.psqt_rel.weight.data[:HKV3_NUM_REL_FEATURES] = p_rel.to(dev0)
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


def _to_cpu_recursive(obj):
    """Chuyển đệ quy mọi Tensor (bao gồm cả XLA/TPU tensor hoặc CUDA tensor) về CPU trước khi lưu .pt."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_recursive(v) for v in obj)
    return obj


def tpu_mark_step():
    """Kích hoạt thực thi bước tính toán bất đồng bộ trên TPU (hỗ trợ cả torch_tpu native và torch_xla)."""
    if HAS_NATIVE_TPU and hasattr(torch, "tpu"):
        if hasattr(torch.tpu, "mark_step"):
            torch.tpu.mark_step()
    elif HAS_XLA:
        xm.mark_step()


def save_portable_checkpoint(data: dict, path: Path, is_tpu: bool = False, is_master: bool = True):
    """Lưu checkpoint .pt tương thích 100% giữa TPU v5e-1/v5e-8, GPU CUDA và CPU."""
    if is_tpu:
        tpu_mark_step()
    if is_master:
        torch.save(_to_cpu_recursive(data), path)


def get_autocast_ctx(device_type: str, enabled: bool):
    """Ngữ cảnh Mixed Precision: Native bfloat16 trên TPU v5e (MXU) / Ampere+ GPU và float16 trên GPU T4x2."""
    if not enabled:
        return contextlib.nullcontext()
    if device_type in ("tpu", "xla"):
        try:
            return torch.autocast(device_type, dtype=torch.bfloat16, enabled=True)
        except Exception:
            return contextlib.nullcontext()
    if device_type == "cuda":
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        return torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
    return contextlib.nullcontext()


def step_optimizer(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    clip_params=None,
    max_norm: Optional[float] = None,
    base_model: Optional[HalfKPv3Network] = None,
    is_tpu: bool = False,
    xla_ddp: bool = False,
):
    """
    Thực hiện backward + gradient clipping + optimizer step + weight clamping:
      - Trên TPU v5e-8 (XLA Multi-Core DDP): Dùng xm.optimizer_step(optimizer, barrier=True) để AllReduce qua ICI 400 GB/s.
      - Trên TPU v5e-1 (Native torch_tpu / XLA bfloat16): Không cần GradScaler, gọi tpu_mark_step() cuối bước.
      - Trên CUDA (T4x2 FP16): Sử dụng GradScaler tiêu chuẩn.
    """
    if is_tpu:
        loss.backward()
        if clip_params is not None and max_norm is not None:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_norm)
        if xla_ddp and HAS_XLA:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
        if base_model is not None:
            base_model.clamp_quantizable_weights()
        if not (xla_ddp and HAS_XLA):
            tpu_mark_step()
    else:
        scaler.scale(loss).backward()
        if clip_params is not None and max_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()
        if base_model is not None:
            base_model.clamp_quantizable_weights()


def _detect_xla_tpu_chip_count() -> int:
    """Phát hiện số lượng chip TPU khả dụng khi chạy dưới runtime torch_xla (1 cho v5e-1, 8 cho v5e-8)."""
    if not HAS_XLA:
        return 0
    try:
        if xr is not None and hasattr(xr, "global_runtime_device_count"):
            return int(xr.global_runtime_device_count())
    except Exception:
        pass
    try:
        devs = xm.get_xla_supported_devices()
        return len(devs) if devs else 1
    except Exception:
        return 1


def _xla_mp_entry(rank: int, args, shared_tensors):
    """Hàm entry point cho từng tiến trình TPU khi chạy 8-chip Data-Parallel trên TPU v5e-8 (xmp.spawn)."""
    _run_training_pipeline(args, preloaded_data=shared_tensors, xla_rank=rank)


# ═══════════════════════════════════════════════════════════════════════════
# 8. VÒNG LẶP HUẤN LUYỆN CHÍNH (TỰ ĐỘNG TỐI ƯU KAGGLE T4x2, TPU v5e-1 & TPU v5e-8)
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="HalfKPv3 (HKV3) + HARENN Unified Training Pipeline (Auto-Optimized for T4x2, TPU v5e-1 & TPU v5e-8)"
    )
    parser.add_argument("--data", type=str, default="Neuron Network/data", help="Thư mục chứa các file .jsonl/.json")
    parser.add_argument("--cache", type=str, default="", help="Đường dẫn file cache .npz (tùy chọn)")
    parser.add_argument("--sync-hf", action="store_true", help="Tự động tải toàn bộ .jsonl từ Hugging Face Bucket trước khi train")
    parser.add_argument("--hf-bucket", type=str, default="australia1/myrandombucket", help="ID của Hugging Face Bucket")
    parser.add_argument("--epochs", type=int, default=16, help="Số epoch huấn luyện đa nhiệm Giai đoạn 1")
    parser.add_argument("--finetune-eval-epochs", type=int, default=4, help="Số epoch tinh chỉnh chuyên sâu Eval ở Giai đoạn 2")
    parser.add_argument("--batch-size", type=int, default=0, help="Kích thước batch (0 = Tự động theo phần cứng: 16384 cho T4x2, 32768 cho v5e-1, 65536 cho v5e-8)")
    parser.add_argument("--lr", type=float, default=0.0, help="Learning rate cho AdamW (0.0 = Tự động theo phần cứng: 1.5e-3 cho T4x2, 2.5e-3 cho v5e-1, 3.0e-3 cho v5e-8)")
    parser.add_argument("--wdl-weight", type=float, default=0.75, help="Tỷ trọng WDL BCE Loss so với Huber Loss (mặc định: 0.75)")
    parser.add_argument("--symmetry", type=int, default=1, choices=[1, 2], help="Chế độ đối xứng lật màu dọc (mặc định: 1x do kiến trúc đã đối xứng đại số tuyệt đối)")
    parser.add_argument("--feature-dropout", type=float, default=0.0, help="Tỷ lệ dropout chỉ riêng trên bảng W_full ở Stage 1 (mặc định: 0.0 - tắt hoàn toàn)")
    parser.add_argument("--val-split", type=float, default=0.03, help="Tỷ lệ tập validation")
    parser.add_argument("--tpu", action="store_true", help="Ưu tiên sử dụng thiết bị Google Cloud / Colab / Kaggle TPU (v5e-1 / v5e-8)")
    parser.add_argument("--tpu-cores", type=int, default=0, help="Số lõi TPU khi chạy XLA đa tiến trình (0 = Tự động nhận diện v5e-1 hoặc v5e-8, 1 = Đơn tiến trình)")
    parser.add_argument("--no-qat", action="store_true", help="Tắt Quantization-Aware Training")
    parser.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision (BF16 trên TPU / FP16 trên GPU)")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn checkpoint (.pt) hoặc file nhị phân (.harenn) để tiếp tục train")
    parser.add_argument("--resume-epoch", type=int, default=None, help="Chỉ định epoch bắt đầu (mặc định: tự động nhận diện từ checkpoint + 1)")
    parser.add_argument("--reset-optimizer", action="store_true", help="Khởi tạo lại optimizer & LR schedule, chỉ dùng weights của model cũ")
    parser.add_argument("--skip-stage1", action="store_true", help="Bỏ qua Giai đoạn 1 và chuyển thẳng sang Stage 2 Fine-tune")
    parser.add_argument("--finetune-tactical", action="store_true", help="Chỉ finetune 3 đầu chiến thuật (tau, rho, rs), đóng băng toàn bộ backbone và eval_head")
    parser.add_argument("--tactical-epochs", type=int, default=2, help="Số epochs finetune đầu chiến thuật (mặc định: 2)")
    parser.add_argument("--output", type=str, default="Neuron Network/nextfish_hkv3.harenn", help="Đường dẫn xuất file .harenn")
    args = parser.parse_args()

    # Nếu chạy trên TPU v5e-8 với torch_xla từ dòng lệnh và có >1 chip:
    # Nạp dữ liệu 1 lần duy nhất vào Shared Memory RAM của tiến trình cha trước khi gọi xmp.spawn
    in_ipython = ("ipykernel" in sys.modules) or ("google.colab" in sys.modules)
    if (not HAS_NATIVE_TPU) and HAS_XLA and (not torch.cuda.is_available()) and (not in_ipython) and args.tpu_cores != 1:
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        n_chips = args.tpu_cores if args.tpu_cores > 1 else _detect_xla_tpu_chip_count()
        if n_chips > 1:
            print(f"[TPU v5e-{n_chips} Auto-Detect] Phát hiện {n_chips} chip TPU XLA! Đang nạp dữ liệu vào Shared Memory RAM (Zero-Copy)...", flush=True)
            if args.sync_hf:
                sync_from_huggingface(args.hf_bucket, args.data)
            us_d, them_d, ev_d, tau_d, rho_d, rs_d = load_or_build_hkv3_cache(
                args.data, cache_path=args.cache if args.cache else None, use_cache=True
            )
            shared_tensors = (
                torch.from_numpy(us_d).share_memory_(),
                torch.from_numpy(them_d).share_memory_(),
                torch.from_numpy(ev_d).share_memory_(),
                torch.from_numpy(tau_d).share_memory_(),
                torch.from_numpy(rho_d).share_memory_(),
                torch.from_numpy(rs_d).share_memory_(),
            )
            print(f"[TPU v5e-{n_chips} Spawn] Khởi chạy {n_chips} tiến trình huấn luyện song song qua liên kết ICI...", flush=True)
            xmp.spawn(_xla_mp_entry, args=(args, shared_tensors), nprocs=n_chips, start_method="fork")
            return

    _run_training_pipeline(args, preloaded_data=None, xla_rank=0)


def _run_training_pipeline(args, preloaded_data=None, xla_rank: int = 0):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    is_tpu = False
    num_tpu_chips = 0
    xla_world_size = 1
    xla_ddp = False

    if args.tpu or HAS_NATIVE_TPU or (HAS_XLA and not torch.cuda.is_available()):
        if HAS_NATIVE_TPU:
            try:
                num_tpu_chips = int(torch.tpu.device_count()) if hasattr(torch.tpu, "device_count") else 1
                device = torch.device("tpu:0" if num_tpu_chips > 1 else "tpu")
                _ = torch.zeros(1, device=device)
                is_tpu = True
            except Exception as e:
                if args.tpu and not HAS_XLA:
                    raise RuntimeError(f"Không thể khởi tạo thiết bị Native TPU (torch_tpu): {e}") from e
        if not is_tpu and HAS_XLA:
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            try:
                device = xm.xla_device()
                xla_world_size = int(xm.xrt_world_size()) if hasattr(xm, "xrt_world_size") else 1
                xla_rank = int(xm.get_ordinal()) if hasattr(xm, "get_ordinal") else xla_rank
                xla_ddp = (xla_world_size > 1)
                num_tpu_chips = xla_world_size if xla_ddp else max(1, _detect_xla_tpu_chip_count())
                is_tpu = True
            except Exception as e:
                if args.tpu:
                    raise RuntimeError(f"Không thể khởi tạo thiết bị TPU (torch_xla): {e}") from e
                device = torch.device("cpu")
        if not is_tpu:
            if args.tpu:
                raise RuntimeError("Cờ --tpu được bật nhưng không tìm thấy torch_tpu hoặc torch_xla khả dụng!")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = (not xla_ddp) or (xla_rank == 0)
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    use_amp = (is_tpu or device.type == "cuda") and (not args.no_amp)
    dev_type = device.type

    # Tự động chọn Batch Size & Learning Rate tối ưu theo phần cứng nếu người dùng để mặc định (0)
    if args.batch_size <= 0:
        if is_tpu:
            if xla_ddp:
                args.batch_size = 16384  # 16384 x 8 cores = 131,072 global batch size trên TPU v5e-8!
            elif num_tpu_chips >= 2:
                args.batch_size = 65536  # Multi-chip Tensor-Parallel trên TPU v5e-8
            else:
                args.batch_size = 32768  # Tối ưu cho TPU v5e-1 (16GB HBM2e)
        elif num_gpus >= 1:
            args.batch_size = 16384      # Tối ưu cho Kaggle T4x2 / GPU
        else:
            args.batch_size = 4096       # CPU fallback

    if args.lr <= 0.0:
        effective_global_bs = args.batch_size * xla_world_size
        if effective_global_bs >= 65536:
            args.lr = 3.0e-3
        elif effective_global_bs >= 32768:
            args.lr = 2.5e-3
        else:
            args.lr = 1.5e-3

    if is_master:
        print("=" * 76)
        print("  HALFKPV3 (HKV3) + HARENN TRAINING PIPELINE (SINGLE BIG NET ~88.85 MB)")
        print("=" * 76)
        if is_tpu:
            tpu_desc = f"TPU v5e-{num_tpu_chips}" if num_tpu_chips >= 1 else "Google Cloud / Colab TPU"
            mode_desc = f"{xla_world_size}x Core ICI Data-Parallel" if xla_ddp else (
                "Multi-Chip ICI Tensor-Parallel" if (HAS_NATIVE_TPU and num_tpu_chips >= 2) else "Single-Chip Unified HBM2e"
            )
            print(f"  Thiết bị TPU:            {tpu_desc} ({device}) | Mode: {mode_desc} | MXU BF16: {'ON' if use_amp else 'OFF'}")
            print("  Tối ưu hóa TPU:          Static Batch Graph + Zero .item() Sync + Native Gather-Sum + Native Huber")
        elif num_gpus > 0:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            amp_str = "BF16" if torch.cuda.is_bf16_supported() else "FP16"
            print(f"  Thiết bị GPU:            {num_gpus}x GPU ({', '.join(gpu_names)}) | AMP {amp_str}: {'ON' if use_amp else 'OFF'}")
        else:
            print("  Thiết bị:                CPU")
        print(f"  Không gian đặc trưng:    {HKV3_NUM_FEATURES:,} (32 Mirrored King Buckets x 11 Channels x 64)")
        print(f"  Virtual Factorizer:      W_full({HKV3_NUM_FEATURES:,}) + W_piece({HKV3_NUM_PIECE_FEATURES}) + W_rel({HKV3_NUM_REL_FEATURES:,})")
        print(f"  Cấu trúc tầng ẩn:        {HKV3_L1_SIZE} -> PairwiseGating({HKV3_L1_PAIR_SIZE}) -> SiameseTwinFC2({HKV3_L2_SIZE}) -> 4 Heads")
        print(f"  Đối xứng dữ liệu:        {args.symmetry}x Data Loader + Bit-Exact Algebraic Parity")
        print(f"  W_full Dropout:          {args.feature_dropout:.2%} (Chỉ áp dụng lên emb_full ở Stage 1, bảo toàn vật chất)")
        print(f"  Batch Size:              {args.batch_size:,}/chip (Global: {args.batch_size * xla_world_size:,}) | LR: {args.lr:.2e} | Epochs: {args.epochs} + {args.finetune_eval_epochs} FT")
        if args.resume:
            print(f"  Resume Model:            {args.resume}")
        print(f"  File xuất nhị phân:      {args.output}")
        print("=" * 76, flush=True)

    # 1. Đồng bộ dữ liệu từ Hugging Face Bucket nếu có cờ --sync-hf
    if preloaded_data is not None:
        us_data, them_data, eval_data, tau_data, rho_data, rs_data = [np.asarray(t) for t in preloaded_data]
    else:
        if args.sync_hf and is_master:
            sync_from_huggingface(args.hf_bucket, args.data)
        if xla_ddp and HAS_XLA:
            xm.rendezvous("sync_hf_done")
        us_data, them_data, eval_data, tau_data, rho_data, rs_data = load_or_build_hkv3_cache(
            args.data, cache_path=args.cache if args.cache else None, use_cache=True
        )

    n_total = len(eval_data)
    raw_val_count = min(100000, max(1000, int(n_total * args.val_split)))
    if is_tpu and n_total > args.batch_size * 2:
        # Căn chỉnh tập Val thành bội số nguyên của batch_size trên TPU để dùng chung 100% đồ thị HLO tĩnh với tập Train
        val_batches = max(1, round(raw_val_count / args.batch_size))
        val_count = min(n_total // 2, val_batches * args.batch_size)
    else:
        val_count = raw_val_count

    perm = np.random.permutation(n_total)
    val_indices = perm[:val_count]
    all_train_indices = perm[val_count:]
    # Nếu chạy 8-chip XLA Data-Parallel trên TPU v5e-8, chia đều tập train cho từng chip
    train_indices = all_train_indices[xla_rank::xla_world_size] if xla_ddp else all_train_indices

    if is_master:
        print(f"[Split] Train: {len(all_train_indices):,} base ({len(all_train_indices) * args.symmetry:,} effective, "
              f"{len(train_indices) * args.symmetry:,}/chip) | Val: {len(val_indices):,} positions.", flush=True)

    train_loader = FastHKV3BatchLoader(
        us_data, them_data, eval_data, tau_data, rho_data, rs_data,
        train_indices, batch_size=args.batch_size, shuffle=True,
        symmetry=args.symmetry, device=device, static_shape=is_tpu
    )
    val_loader = FastHKV3BatchLoader(
        us_data, them_data, eval_data, tau_data, rho_data, rs_data,
        val_indices, batch_size=args.batch_size, shuffle=False,
        symmetry=1, device=device, static_shape=is_tpu
    )

    # 3. Khởi tạo mạng HalfKPv3 trên TPU v5e-1 / v5e-8 hoặc Tensor-Parallel trên 2 GPU (Kaggle T4x2)
    if is_tpu:
        if HAS_NATIVE_TPU and num_tpu_chips >= 2:
            dev0 = torch.device("tpu:0")
            dev1 = torch.device("tpu:1")
            if is_master:
                print(f"[TPU v5e-{num_tpu_chips} Tensor-Parallel] Nhánh Left (1024) -> {dev0} || Nhánh Right (1024) -> {dev1} (ICI High-Speed Link!)", flush=True)
        else:
            dev0 = device
            dev1 = device
            if is_master:
                print(f"[TPU v5e Unified HBM] Toàn bộ mô hình + Shared Factorizer Maps đặt trên {dev0} (Active Chips: {xla_world_size})", flush=True)
    else:
        dev0 = torch.device("cuda:0") if num_gpus >= 1 else torch.device("cpu")
        dev1 = torch.device("cuda:1") if num_gpus >= 2 else dev0
        if num_gpus >= 2 and is_master:
            print(f"[T4x2 Tensor-Parallel] Nhánh Left (1024) -> {dev0} || Nhánh Right (1024) -> {dev1} (Zero PCIe Weight Copy!)", flush=True)

    base_model = HalfKPv3Network(
        qat=not args.no_qat,
        dev0=dev0,
        dev1=dev1,
        full_dropout=args.feature_dropout,
        use_amp=use_amp,
    )

    model: nn.Module = base_model
    criterion = HKV3MultiTaskLoss(wdl_weight=args.wdl_weight).to(dev0)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not is_tpu))

    # Phân nhóm tham số tối ưu:
    # - W_full chịu weight_decay cao hơn (5e-5) để ép các ô Vua hiếm dựa vào W_piece & W_rel
    optimizer = optim.AdamW([
        {"params": list(base_model.left_branch.emb_full.parameters()) + list(base_model.right_branch.emb_full.parameters()), "lr": args.lr * 0.8, "weight_decay": 5e-5},
        {"params": list(base_model.left_branch.emb_piece.parameters()) + list(base_model.right_branch.emb_piece.parameters()), "lr": args.lr, "weight_decay": 1e-5},
        {"params": list(base_model.left_branch.emb_rel.parameters()) + list(base_model.right_branch.emb_rel.parameters()), "lr": args.lr, "weight_decay": 1e-5},
        {"params": [base_model.left_branch.bias, base_model.right_branch.bias], "lr": args.lr, "weight_decay": 0.0},
        {"params": base_model.psqt_full.parameters(), "lr": args.lr * 0.5, "weight_decay": 5e-5},
        {"params": base_model.psqt_piece.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.psqt_rel.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.fc2_self.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.fc2_opp.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        {"params": base_model.eval_head.parameters(), "lr": args.lr * 1.2, "weight_decay": 1e-6},
        {"params": base_model.tac_head.parameters(), "lr": args.lr * 1.2, "weight_decay": 1e-6},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=2e-5)

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_pt_path = Path(args.output).with_suffix(".best.pt")

    # 4. Nạp checkpoint / mô hình cũ nếu có tham số --resume
    if args.resume:
        ckpt_dict, loaded_epoch = load_hkv3_weights_or_checkpoint(base_model, args.resume, dev0, dev1)
        if args.resume_epoch is not None:
            start_epoch = args.resume_epoch
            print(f"[Resume] Ghi đè epoch bắt đầu: {start_epoch}", flush=True)
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
            if "scaler_state_dict" in ckpt_dict and use_amp and not is_tpu:
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

    metric_keys = ("total", "eval", "tau", "rho", "rs", "mae_cp", "wdl_err")

    def evaluate_validation() -> Dict[str, float]:
        model.eval()
        totals_t = {k: torch.zeros((), dtype=torch.float32, device=dev0) for k in metric_keys}
        steps = 0
        with torch.no_grad():
            for us_b, them_b, targets_b in val_loader:
                with get_autocast_ctx(dev_type, use_amp):
                    outputs_b = model(us_b, them_b)
                    _, m = criterion(outputs_b, targets_b)
                for k in metric_keys:
                    totals_t[k] = totals_t[k] + m[k]
                steps += 1
                if is_tpu:
                    tpu_mark_step()
        denom = float(max(1, steps))
        return {k: float(totals_t[k].item()) / denom for k in metric_keys}

    # ─── GIAI ĐOẠN ĐẶC BIỆT: CHUYÊN SÂU FINETUNE ĐẦU CHIẾN THUẬT (TAU, RHO, RS) ───
    if args.finetune_tactical:
        if is_master:
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
            tac_loss_sum_t = torch.zeros((), dtype=torch.float32, device=dev0)
            steps = 0
            for us_b, them_b, targets_b in train_loader:
                tac_opt.zero_grad(set_to_none=True)
                with get_autocast_ctx(dev_type, use_amp):
                    outputs_b = model(us_b, them_b)
                    loss_tau = F.binary_cross_entropy_with_logits(outputs_b["tau"], targets_b["tau"])
                    loss_rho = F.binary_cross_entropy_with_logits(outputs_b["rho"], targets_b["rho"])
                    loss_rs = F.binary_cross_entropy_with_logits(outputs_b["rs"], targets_b["rs"])
                    loss = loss_tau + loss_rho + loss_rs

                tac_loss_sum_t = tac_loss_sum_t + loss.detach().float()
                step_optimizer(
                    loss, tac_opt, scaler,
                    clip_params=base_model.tac_head.parameters(),
                    max_norm=1.0,
                    base_model=None,
                    is_tpu=is_tpu,
                    xla_ddp=xla_ddp,
                )
                steps += 1

            tac_sched.step()
            tac_loss_avg = float(tac_loss_sum_t.item()) / max(1, steps)
            dt = time.time() - t0
            val_m = evaluate_validation()
            if is_master:
                print(
                    f"[Tactical Ep {ep:02d}/{args.tactical_epochs:02d} ({dt:5.1f}s)] "
                    f"Train BCE Loss: {tac_loss_avg:.4f} | "
                    f"Val: [tau:{val_m['tau']:.4f} rho:{val_m['rho']:.4f} rs:{val_m['rs']:.4f}] | Val MAE: {val_m['mae_cp']:.2f}cp",
                    flush=True,
                )

        if is_master:
            print(f"\n[Export] Xuất nhị phân cập nhật Tactical Heads: {args.output}...", flush=True)
            export_hkv3_binary(base_model, args.output)
        for p in base_model.parameters():
            p.requires_grad = True

        if is_master:
            print(f"[Done] Tactical heads finetune hoàn tất! -> {args.output}", flush=True)
        return

    # ─── GIAI ĐOẠN 1: HUẤN LUYỆN ĐA NHIỆM ĐỒNG THỜI (EVAL + TAU + RHO + RS) ───
    if args.skip_stage1:
        if is_master:
            print("\n[Stage 1] Bỏ qua Giai đoạn 1 do cờ --skip-stage1 được kích hoạt.", flush=True)
    elif start_epoch > args.epochs:
        if is_master:
            print(
                f"\n[Stage 1] Checkpoint đã hoàn thành {start_epoch - 1}/{args.epochs} epochs. "
                f"Bỏ qua Stage 1 để chuyển tiếp sang Stage 2 (hoặc chỉ định --epochs lớn hơn để train tiếp).",
                flush=True,
            )
    else:
        if is_master:
            print(f"\n[Stage 1] Bắt đầu huấn luyện Đa nhiệm HalfKPv3 (Epoch {start_epoch:02d} -> {args.epochs:02d})...", flush=True)
            if is_tpu:
                print("[TPU Warmup] Batch đầu tiên của Epoch 1 sẽ biên dịch đồ thị XLA HLO tĩnh (~30-45s), sau đó chạy tốc độ tối đa!", flush=True)
        train_keys = ("total", "eval", "tau", "rho", "rs", "mae_cp")
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            train_totals_t = {k: torch.zeros((), dtype=torch.float32, device=dev0) for k in train_keys}
            steps = 0

            for us_b, them_b, targets_b in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with get_autocast_ctx(dev_type, use_amp):
                    outputs_b = model(us_b, them_b)
                    loss, m = criterion(outputs_b, targets_b)

                for k in train_keys:
                    train_totals_t[k] = train_totals_t[k] + m[k]

                step_optimizer(
                    loss, optimizer, scaler,
                    clip_params=base_model.parameters(),
                    max_norm=1.0,
                    base_model=base_model,
                    is_tpu=is_tpu,
                    xla_ddp=xla_ddp,
                )
                steps += 1

            scheduler.step()
            denom = float(max(1, steps))
            train_totals = {k: float(train_totals_t[k].item()) / denom for k in train_keys}
            dt = time.time() - t0

            val_m = evaluate_validation()
            cur_lr = scheduler.get_last_lr()[0]
            pos_per_sec = int((steps * args.batch_size * xla_world_size) / max(1e-3, dt))

            if is_master:
                print(
                    f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:5.1f}s | {pos_per_sec:,} pos/s) | "
                    f"Train Loss: {train_totals['total']:.4f} (MAE: {train_totals['mae_cp']:.1f}cp) | "
                    f"Val Loss: {val_m['total']:.4f} (MAE: {val_m['mae_cp']:.1f}cp, WDL^2.5: {val_m['wdl_err']:.5f}) | "
                    f"[tau:{val_m['tau']:.4f} rho:{val_m['rho']:.4f} rs:{val_m['rs']:.4f}] | LR: {cur_lr:.2e}",
                    flush=True,
                )

            # Cập nhật best_val_loss / best_val_mae TRƯỚC khi đóng gói ckpt_data để lưu đúng kỷ lục mới nhất
            is_new_best = val_m["total"] < best_val_loss
            if is_new_best:
                best_val_loss = val_m["total"]
                best_val_mae = val_m["mae_cp"]

            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
            }
            ckpt_path = Path(args.output).with_suffix(f".epoch{epoch}.pt")
            latest_path = Path(args.output).with_suffix(".latest.pt")
            save_portable_checkpoint(ckpt_data, ckpt_path, is_tpu=is_tpu, is_master=is_master)
            save_portable_checkpoint(ckpt_data, latest_path, is_tpu=is_tpu, is_master=is_master)

            if is_new_best and is_master:
                save_portable_checkpoint(ckpt_data, best_pt_path, is_tpu=is_tpu, is_master=True)
                export_hkv3_binary(base_model, args.output)
            if xla_ddp and HAS_XLA:
                xm.rendezvous(f"ep_{epoch}_save_done")

    # ─── GIAI ĐOẠN 2: TINH CHỈNH CHUYÊN SÂU ĐẦU ĐỌC EVAL (2A: JOINT REFINEMENT + 2B: LOCKED HEAD) ───
    if args.finetune_eval_epochs > 0:
        phase_a = max(1, args.finetune_eval_epochs // 2)
        phase_b = max(1, args.finetune_eval_epochs - phase_a)
        if is_master:
            print(f"\n[Stage 2] Tinh chỉnh chuyên sâu Eval ({phase_a} epochs Phase 2A + {phase_b} epochs Phase 2B)...", flush=True)

        # Tắt hoàn toàn W_full dropout ở Stage 2 để mạng hội tụ chính xác từng centipawn
        base_model.set_full_dropout(0.0)

        # Khôi phục Best Model từ Stage 1 trước khi bước vào Stage 2A (tránh bắt đầu từ epoch cuối bị overfit)
        if best_pt_path.exists():
            if is_master:
                print(f"[Stage 2A] Nạp lại Best Checkpoint từ Stage 1 ({best_pt_path})...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, str(best_pt_path), dev0, dev1)
        elif Path(args.output).exists():
            if is_master:
                print(f"[Stage 2A] Nạp lại Best Binary từ Stage 1 ({args.output})...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, args.output, dev0, dev1)

        # Nếu chưa có mốc best_val_mae (ví dụ khi chạy --skip-stage1 từ file .harenn), đo ngay trên tập Val
        if not math.isfinite(best_val_mae):
            init_val = evaluate_validation()
            best_val_loss = init_val["total"]
            best_val_mae = init_val["mae_cp"]
            if is_master:
                print(f"[Stage 2A] Baseline Val MAE trước khi fine-tune: {best_val_mae:.2f} cp", flush=True)

        # Phase 2A: Ưu tiên Eval gấp 4 lần nhưng vẫn giữ neo tau/rho/rs
        ft_opt_a = optim.AdamW([
            {"params": list(base_model.left_branch.emb_full.parameters()) + list(base_model.right_branch.emb_full.parameters()), "lr": 1.0e-4, "weight_decay": 5e-5},
            {"params": list(base_model.left_branch.emb_piece.parameters()) + list(base_model.right_branch.emb_piece.parameters()), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": list(base_model.left_branch.emb_rel.parameters()) + list(base_model.right_branch.emb_rel.parameters()), "lr": 1.5e-4, "weight_decay": 1e-5},
            {"params": [base_model.left_branch.bias, base_model.right_branch.bias], "lr": 1.0e-4, "weight_decay": 0.0},
            {"params": base_model.psqt_full.parameters(), "lr": 1.0e-4, "weight_decay": 5e-5},
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
            for us_b, them_b, targets_b in train_loader:
                ft_opt_a.zero_grad(set_to_none=True)
                with get_autocast_ctx(dev_type, use_amp):
                    outputs_b = model(us_b, them_b)
                    loss_ev = criterion.compute_eval_loss(outputs_b["eval"], targets_b["eval"])
                    loss_aux = (
                        F.binary_cross_entropy_with_logits(outputs_b["tau"], targets_b["tau"])
                        + F.binary_cross_entropy_with_logits(outputs_b["rho"], targets_b["rho"])
                        + F.binary_cross_entropy_with_logits(outputs_b["rs"], targets_b["rs"])
                    )
                    loss = 4.0 * loss_ev + 0.15 * loss_aux
                step_optimizer(
                    loss, ft_opt_a, scaler,
                    clip_params=base_model.parameters(),
                    max_norm=0.5,
                    base_model=base_model,
                    is_tpu=is_tpu,
                    xla_ddp=xla_ddp,
                )

            ft_sched_a.step()
            val_m = evaluate_validation()
            dt = time.time() - t0
            if is_master:
                print(
                    f"[Phase 2A {ep:02d}/{phase_a:02d} ({dt:5.1f}s)] "
                    f"Val Eval Loss: {val_m['eval']:.4f} | Val MAE: {val_m['mae_cp']:.2f}cp | WDL^2.5: {val_m['wdl_err']:.5f}",
                    flush=True,
                )
            if val_m["mae_cp"] < best_val_mae:
                best_val_mae = val_m["mae_cp"]
                if is_master:
                    save_portable_checkpoint({
                        "stage": "2A_best",
                        "model_state_dict": base_model.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_mae": best_val_mae,
                    }, best_pt_path, is_tpu=is_tpu, is_master=True)
                    export_hkv3_binary(base_model, args.output)
            if xla_ddp and HAS_XLA:
                xm.rendezvous(f"p2a_{ep}_done")

        ckpt_2a_path = Path(args.output).with_suffix(".phase2a.pt")
        save_portable_checkpoint({
            "stage": "2A",
            "model_state_dict": base_model.state_dict(),
            "best_val_mae": best_val_mae,
        }, ckpt_2a_path, is_tpu=is_tpu, is_master=is_master)
        if xla_ddp and HAS_XLA:
            xm.rendezvous("p2a_complete")

        # Khôi phục Best Model sau Phase 2A trước khi đóng băng Backbone sang Phase 2B
        if best_pt_path.exists():
            if is_master:
                print(f"[Stage 2B] Nạp lại Best Checkpoint ({best_pt_path}, MAE: {best_val_mae:.2f} cp) trước khi khóa Backbone...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, str(best_pt_path), dev0, dev1)
        elif Path(args.output).exists():
            if is_master:
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
            for us_b, them_b, targets_b in train_loader:
                ft_opt_b.zero_grad(set_to_none=True)
                with get_autocast_ctx(dev_type, use_amp):
                    outputs_b = model(us_b, them_b)
                    loss_ev = criterion.compute_eval_loss(outputs_b["eval"], targets_b["eval"])
                step_optimizer(
                    loss_ev, ft_opt_b, scaler,
                    clip_params=None,
                    max_norm=None,
                    base_model=base_model,
                    is_tpu=is_tpu,
                    xla_ddp=xla_ddp,
                )

            ft_sched_b.step()
            val_m = evaluate_validation()
            dt = time.time() - t0
            if is_master:
                print(
                    f"[Phase 2B {ep:02d}/{phase_b:02d} ({dt:5.1f}s)] "
                    f"Val Eval Loss: {val_m['eval']:.4f} | Val MAE: {val_m['mae_cp']:.2f}cp | WDL^2.5: {val_m['wdl_err']:.5f}",
                    flush=True,
                )
            if val_m["mae_cp"] < best_val_mae:
                best_val_mae = val_m["mae_cp"]
                if is_master:
                    save_portable_checkpoint({
                        "stage": "2B_best",
                        "model_state_dict": base_model.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_mae": best_val_mae,
                    }, best_pt_path, is_tpu=is_tpu, is_master=True)
                    export_hkv3_binary(base_model, args.output)
            if xla_ddp and HAS_XLA:
                xm.rendezvous(f"p2b_{ep}_done")

        ckpt_2b_path = Path(args.output).with_suffix(".phase2b.pt")
        save_portable_checkpoint({
            "stage": "2B",
            "model_state_dict": base_model.state_dict(),
            "best_val_mae": best_val_mae,
        }, ckpt_2b_path, is_tpu=is_tpu, is_master=is_master)

        for p in base_model.parameters():
            p.requires_grad = True

    # ─── KIỂM CHỨNG ĐỐI XỨNG SỐ NGUYÊN C++ AVX2 SAU KHI XUẤT FILE ───
    if is_master:
        # Nạp lại chính xác Best Model đã lưu để tránh model ở epoch cuối ghi đè làm mất best model!
        if best_pt_path.exists():
            print(f"\n[Verification] Nạp lại Best Model (MAE: {best_val_mae:.2f} cp) từ {best_pt_path} để kiểm chứng...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, str(best_pt_path), dev0, dev1)
        elif Path(args.output).exists():
            print(f"\n[Verification] Nạp lại Best Model (MAE: {best_val_mae:.2f} cp) từ {args.output} để kiểm chứng...", flush=True)
            _, _ = load_hkv3_weights_or_checkpoint(base_model, args.output, dev0, dev1)
        quant_tensors = export_hkv3_binary(base_model, args.output)

        # Chuyển model về CPU cho bước kiểm chứng 64 thế cờ cuối cùng (tránh XLA biên dịch lại đồ thị cho batch=64)
        base_model.to("cpu")
        base_model.dev0 = torch.device("cpu")
        base_model.dev1 = torch.device("cpu")

        sample_n = min(64, len(val_indices))
        verify_integer_avx2_parity_and_accuracy(
            quant_tensors,
            us_data[val_indices[:sample_n]],
            them_data[val_indices[:sample_n]],
            model=base_model,
        )
        print(f"\n[Done] Huấn luyện hoàn tất! Best Validation MAE: {best_val_mae:.2f} cp -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
