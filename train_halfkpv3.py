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
import glob
import json
import math
import os
import random
import struct
import sys
import time
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


def fen_to_hkv3_sparse_pair(fen: str, stm_override: Optional[int] = None) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
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
        stm = int(stm_override)
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

    def process_record(pos: dict):
        fen = pos.get("fen", "")
        if not fen:
            return
        board_key = fen.split()[0] + " " + (fen.split()[1] if len(fen.split()) > 1 else "w")
        h = hash(board_key)

        parsed = fen_to_hkv3_sparse_pair(fen, pos.get("stm"))
        if parsed is None:
            return
        us_arr, them_arr, _ = parsed

        # Lưu ý: Trong dữ liệu .jsonl sinh bởi parallel_generate.py, eval_score ĐÃ LÀ điểm
        # theo góc nhìn bên đi (side-to-move / 'us').
        ev_cp = float(pos.get("eval_score", 0.0))
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

    try:
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for pos in data.get("positions", []):
                if isinstance(pos, dict):
                    process_record(pos)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        pos = json.loads(line)
                        if isinstance(pos, dict):
                            process_record(pos)
                    except Exception:
                        continue
    except Exception:
        pass

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
    else:
        cpath = data_path / "_hkv3_sparse_cache_v1.npz"

    if cpath.exists() and use_cache:
        t0 = time.time()
        print(f"[Cache] Đang nạp cache nhị phân HKV3 từ {cpath}...", flush=True)
        try:
            data = np.load(str(cpath), allow_pickle=False)
            us_indices = data["us_indices"]
            them_indices = data["them_indices"]
            labels_eval = data["labels_eval"]
            labels_tau = data["labels_tau"]
            labels_rho = data["labels_rho"]
            labels_rs = data["labels_rs"]
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
    if not valid_results:
        raise RuntimeError("Không trích xuất được thế cờ hợp lệ nào từ các file dữ liệu!")

    all_hashes = np.concatenate([r[0] for r in valid_results], axis=0)
    all_us = np.concatenate([r[1] for r in valid_results], axis=0)
    all_them = np.concatenate([r[2] for r in valid_results], axis=0)
    all_eval = np.concatenate([r[3] for r in valid_results], axis=0)
    all_tau = np.concatenate([r[4] for r in valid_results], axis=0)
    all_rho = np.concatenate([r[5] for r in valid_results], axis=0)
    all_rs = np.concatenate([r[6] for r in valid_results], axis=0)

    total_raw = len(all_eval)
    _, unique_idx = np.unique(all_hashes, return_index=True)
    unique_idx.sort()

    us_indices = all_us[unique_idx]
    them_indices = all_them[unique_idx]
    labels_eval = all_eval[unique_idx]
    labels_tau = all_tau[unique_idx]
    labels_rho = all_rho[unique_idx]
    labels_rs = all_rs[unique_idx]

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
# 4. BỘ NẠP BATCH SIÊU TỐC TRÊN GPU VỚI ĐỐI XỨNG LẬT MÀU 2X (SYMMETRY=2)
# ═══════════════════════════════════════════════════════════════════════════
class FastHKV3BatchLoader:
    """
    Bộ nạp dữ liệu trực tiếp từ bộ nhớ NumPy sang GPU với:
      - Đối xứng lật màu dọc 2x (symmetry=2): Hoán vị (us, them) <-> (them, us) và đảo dấu eval -> -eval.
      - Feature Dropout (nhiễu bàn cờ trên các quân không phải Vua để chống overfitting ở bảng W_full).
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
        symmetry: int = 2,
        feature_dropout: float = 0.0,
        device: torch.device = torch.device("cuda"),
    ):
        self.us_data = us_data
        self.them_data = them_data
        self.eval_data = eval_data
        self.tau_data = tau_data
        self.rho_data = rho_data
        self.rs_data = rs_data
        self.base_indices = np.asarray(indices, dtype=np.int64)
        self.num_base = len(self.base_indices)
        self.symmetry = symmetry if symmetry in (1, 2) else 2
        self.total_samples = self.num_base * self.symmetry
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_dropout = feature_dropout
        self.device = device

    def __len__(self) -> int:
        return (self.total_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.total_samples)
        else:
            perm = np.arange(self.total_samples, dtype=np.int64)

        for start in range(0, self.total_samples, self.batch_size):
            v_ids = perm[start:start + self.batch_size]
            is_flipped = (v_ids >= self.num_base)
            raw_pos = v_ids % self.num_base
            actual_ids = self.base_indices[raw_pos]

            us_np = self.us_data[actual_ids]
            them_np = self.them_data[actual_ids]
            ev_np = self.eval_data[actual_ids]

            if self.symmetry == 2 and np.any(is_flipped):
                us_np = np.where(is_flipped[:, None], self.them_data[actual_ids], us_np)
                them_np = np.where(is_flipped[:, None], self.us_data[actual_ids], them_np)
                ev_np = np.where(is_flipped, -ev_np, ev_np)

            us_t = torch.from_numpy(us_np.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
            them_t = torch.from_numpy(them_np.astype(np.int64, copy=False)).to(self.device, non_blocking=True)

            if self.feature_dropout > 0.0 and self.shuffle:
                pad_val = PADDING_FEATURE_IDX
                drop_u = (torch.rand_like(us_t, dtype=torch.float32) < self.feature_dropout) & (us_t < pad_val)
                drop_v = (torch.rand_like(them_t, dtype=torch.float32) < self.feature_dropout) & (them_t < pad_val)
                us_t = torch.where(drop_u, pad_val, us_t)
                them_t = torch.where(drop_v, pad_val, them_t)

            targets = {
                "eval": torch.from_numpy(ev_np.copy()).to(self.device, non_blocking=True),
                "tau": torch.from_numpy(self.tau_data[actual_ids]).to(self.device, non_blocking=True),
                "rho": torch.from_numpy(self.rho_data[actual_ids]).to(self.device, non_blocking=True),
                "rs": torch.from_numpy(self.rs_data[actual_ids]).to(self.device, non_blocking=True),
            }
            yield us_t, them_t, targets


# ═══════════════════════════════════════════════════════════════════════════
# 5. KIẾN TRÚC MẠNG NƠ-RON ĐA NHIỆM HALFKPV3 (HKV3) CHUẨN ĐẠI SỐ & SIMD
# ═══════════════════════════════════════════════════════════════════════════
def fake_quant_ste(x: torch.Tensor, scale: float, min_q: float, max_q: float) -> torch.Tensor:
    """Straight-Through Estimator (STE) mô phỏng lượng tử hóa số nguyên C++ trong quá trình train."""
    q = torch.clamp(torch.round(x * scale), min=min_q, max=max_q) / scale
    return x + (q - x).detach()


class HalfKPv3BranchL1(nn.Module):
    """
    Một nửa nhánh tích lũy Tầng 1 (1024 chiều: Left hoặc Right của Pairwise Bilinear Gating).
    Thiết kế tách nhánh này cho phép chạy Tensor-Parallel tự nhiên trên Kaggle T4x2:
      - GPU 0 (cuda:0) giữ nhánh Left  (cột 0..1023)
      - GPU 1 (cuda:1) giữ nhánh Right (cột 1024..2047)
    Triệt tiêu 100% chi phí đồng bộ trọng số 184 MB qua PCIe của DataParallel truyền thống!
    """

    def __init__(self, piece_map_pad: torch.Tensor, rel_map_pad: torch.Tensor, qat: bool = True):
        super().__init__()
        self.qat = qat
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
        us_p = self.piece_map_pad[us_idx]
        us_r = self.rel_map_pad[us_idx]
        acc_us = self.emb_full(us_idx) + self.emb_piece(us_p) + self.emb_rel(us_r) + self.bias

        them_p = self.piece_map_pad[them_idx]
        them_r = self.rel_map_pad[them_idx]
        acc_them = self.emb_full(them_idx) + self.emb_piece(them_p) + self.emb_rel(them_r) + self.bias

        c_us = torch.clamp(acc_us, 0.0, 1.0)
        c_them = torch.clamp(acc_them, 0.0, 1.0)
        if self.qat and self.training:
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
      - Pairwise Bilinear Gating: 2048 -> 1024 per perspective
      - Siamese Perspective Twin FC2: W_self(256x1024) & W_opp(256x1024)
      - Orthogonal Parity Readout Heads: eval_head (Odd, No Bias) & tac_head (Even, With Bias)
    """

    def __init__(self, qat: bool = True, dev0: torch.device = torch.device("cpu"), dev1: Optional[torch.device] = None):
        super().__init__()
        self.qat = qat
        self.dev0 = dev0
        self.dev1 = dev1 if dev1 is not None else dev0

        piece_map_pad, rel_map_pad = build_hkv3_canonical_factorizer_maps()
        self.register_buffer("piece_map_pad", piece_map_pad.to(self.dev0))
        self.register_buffer("rel_map_pad", rel_map_pad.to(self.dev0))

        # ─── 1. Tầng 1: Hai nhánh Left (0..1023) & Right (1024..2047) của Virtual Factorizer ───
        self.left_branch = HalfKPv3BranchL1(piece_map_pad, rel_map_pad, qat=qat).to(self.dev0)
        self.right_branch = HalfKPv3BranchL1(piece_map_pad, rel_map_pad, qat=qat).to(self.dev1)

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
            us_idx_1 = us_idx.to(self.dev1, non_blocking=True)
            them_idx_1 = them_idx.to(self.dev1, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=us_idx.is_cuda):
                u_right, v_right = self.right_branch(us_idx_1, them_idx_1)
            u_left, v_left = self.left_branch(us_idx, them_idx)
            u_right = u_right.to(self.dev0, non_blocking=True)
            v_right = v_right.to(self.dev0, non_blocking=True)
        else:
            u_left, v_left = self.left_branch(us_idx, them_idx)
            u_right, v_right = self.right_branch(us_idx, them_idx)

        # Bước 2: Pairwise Bilinear Gating (u = u_left * u_right, v = v_left * v_right)
        u = u_left * u_right
        v = v_left * v_right
        if self.qat and self.training:
            u = fake_quant_ste(u, QA_SCALE, 0.0, 127.0)
            v = fake_quant_ste(v, QA_SCALE, 0.0, 127.0)

        psqt_us = self._compute_psqt(us_idx)
        psqt_them = self._compute_psqt(them_idx)

        # Bước 3: Siamese Perspective Twin FC2
        if self.qat and self.training:
            w_self = fake_quant_ste(self.fc2_self.weight, QW_FC2_SCALE, -127.0, 127.0)
            b_fc2 = fake_quant_ste(self.fc2_self.bias, QA_SCALE * QW_FC2_SCALE, -2147483647.0, 2147483647.0)
            w_opp = fake_quant_ste(self.fc2_opp.weight, QW_FC2_SCALE, -127.0, 127.0)
            z_us = F.linear(u, w_self, b_fc2) + F.linear(v, w_opp, None)
            z_them = F.linear(v, w_self, b_fc2) + F.linear(u, w_opp, None)
        else:
            z_us = self.fc2_self(u) + self.fc2_opp(v)
            z_them = self.fc2_self(v) + self.fc2_opp(u)

        # Bước 4: Kích hoạt phi tuyến SCReLU tại tầng 2
        h_us = torch.clamp(z_us, 0.0, 1.0).square()
        h_them = torch.clamp(z_them, 0.0, 1.0).square()

        # Bước 5: Phân rã Trực giao Chẵn - Lẻ ngay sau phi tuyến FC2
        h_odd = h_us - h_them
        h_even = h_us + h_them

        # Bước 6: Đầu đọc + Nhánh tắt PSQT
        psqt_eval = psqt_us[:, 0] - psqt_them[:, 0]
        psqt_tac = psqt_us[:, 1:4] + psqt_them[:, 1:4]

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

    def __init__(self, wdl_weight: float = 0.75, wdl_scale_pawns: float = WDL_SCALE_PAWNS):
        super().__init__()
        self.wdl_weight = wdl_weight
        self.wdl_scale_pawns = wdl_scale_pawns

    def compute_eval_loss(self, pred_pawns: torch.Tensor, target_pawns: torch.Tensor) -> torch.Tensor:
        target_clamped = torch.clamp(target_pawns, -10.0, 10.0)

        # 1. Smooth-L1 Huber Loss trực tiếp trên đơn vị Pawns (beta = 0.5 pawns = 50 cp)
        loss_huber = F.smooth_l1_loss(pred_pawns, target_clamped, reduction="mean", beta=0.5)

        # 2. WDL Sigmoid Binary Cross-Entropy (with logits để an toàn tuyệt đối với FP16 autocast)
        target_wdl = torch.sigmoid(target_clamped / self.wdl_scale_pawns)
        loss_wdl = F.binary_cross_entropy_with_logits(
            pred_pawns / self.wdl_scale_pawns, target_wdl, reduction="mean"
        )

        return self.wdl_weight * loss_wdl + (1.0 - self.wdl_weight) * loss_huber

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_eval = self.compute_eval_loss(outputs["eval"], targets["eval"])

        loss_tau = F.binary_cross_entropy_with_logits(outputs["tau"], targets["tau"], reduction="mean")
        loss_rho = F.binary_cross_entropy_with_logits(outputs["rho"], targets["rho"], reduction="mean")
        loss_rs = F.binary_cross_entropy_with_logits(outputs["rs"], targets["rs"], reduction="mean")

        total_loss = loss_eval + 0.15 * (loss_tau + loss_rho + loss_rs)

        with torch.no_grad():
            mae_cp = torch.mean(torch.abs(outputs["eval"] - targets["eval"])).item() * 100.0
            pred_wdl = torch.sigmoid(outputs["eval"] / self.wdl_scale_pawns)
            tgt_wdl = torch.sigmoid(targets["eval"] / self.wdl_scale_pawns)
            wdl_err = torch.mean(torch.abs(pred_wdl - tgt_wdl) ** 2.5).item()

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
):
    """
    Mô phỏng chính xác 100% đường đi số nguyên C++ AVX2 (_mm256_maddubs_epi16) trên tập mẫu để:
      1. Kiểm chứng không có tràn số bão hòa int16_t trong maddubs (u_prod, v_prod in [0, 127]).
      2. Kiểm chứng định lý Phản đối xứng BIT-EXACT: E_int(us, them) + E_int(them, us) == 0.
      3. Kiểm chứng định lý Đối xứng BIT-EXACT: [tau, rho, rs]_int(us, them) == [tau, rho, rs]_int(them, us).
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


# ═══════════════════════════════════════════════════════════════════════════
# 8. VÒNG LẶP HUẤN LUYỆN CHÍNH (HỖ TRỢ ĐA GPU KAGGLE T4x2 + AMP FP16)
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="HalfKPv3 (HKV3) + HARENN Training Pipeline (Kaggle T4x2 Ready)"
    )
    parser.add_argument("--data", type=str, default="Neuron Network/data", help="Thư mục chứa các file .jsonl/.json")
    parser.add_argument("--cache", type=str, default="", help="Đường dẫn file cache .npz (tùy chọn)")
    parser.add_argument("--sync-hf", action="store_true", help="Tự động tải toàn bộ .jsonl từ Hugging Face Bucket trước khi train")
    parser.add_argument("--hf-bucket", type=str, default="australia1/myrandombucket", help="ID của Hugging Face Bucket")
    parser.add_argument("--epochs", type=int, default=16, help="Số epoch huấn luyện đa nhiệm Giai đoạn 1")
    parser.add_argument("--finetune-eval-epochs", type=int, default=4, help="Số epoch tinh chỉnh chuyên sâu Eval ở Giai đoạn 2")
    parser.add_argument("--batch-size", type=int, default=16384, help="Kích thước batch (khuyến nghị 16384 cho T4x2)")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate cơ sở cho AdamW")
    parser.add_argument("--wdl-weight", type=float, default=0.75, help="Tỷ trọng WDL BCE Loss so với Huber Loss (mặc định: 0.75)")
    parser.add_argument("--symmetry", type=int, default=2, choices=[1, 2], help="Chế độ đối xứng lật màu dọc (mặc định: 2x)")
    parser.add_argument("--feature-dropout", type=float, default=0.03, help="Tỷ lệ nhiễu ẩn quân cờ để chống overfit W_full")
    parser.add_argument("--val-split", type=float, default=0.03, help="Tỷ lệ tập validation")
    parser.add_argument("--no-qat", action="store_true", help="Tắt Quantization-Aware Training")
    parser.add_argument("--no-amp", action="store_true", help="Tắt Mixed Precision (FP16 Tensor Cores)")
    parser.add_argument("--resume", type=str, default="", help="Đường dẫn checkpoint .pt để tiếp tục train")
    parser.add_argument("--output", type=str, default="Neuron Network/nextfish_hkv3.harenn", help="Đường dẫn xuất file .harenn")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    print("=" * 76)
    print("  HALFKPV3 (HKV3) + HARENN TRAINING PIPELINE (SINGLE BIG NET ~88.85 MB)")
    print("=" * 76)
    if num_gpus > 0:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"  Thiết bị GPU:            {num_gpus}x GPU ({', '.join(gpu_names)})")
    else:
        print("  Thiết bị:                CPU")
    print(f"  Không gian đặc trưng:    {HKV3_NUM_FEATURES:,} (32 Mirrored King Buckets x 11 Channels x 64)")
    print(f"  Virtual Factorizer:      W_full({HKV3_NUM_FEATURES:,}) + W_piece({HKV3_NUM_PIECE_FEATURES}) + W_rel({HKV3_NUM_REL_FEATURES:,})")
    print(f"  Cấu trúc tầng ẩn:        {HKV3_L1_SIZE} -> PairwiseGating({HKV3_L1_PAIR_SIZE}) -> SiameseTwinFC2({HKV3_L2_SIZE}) -> 4 Heads")
    print(f"  Đối xứng dữ liệu:        {args.symmetry}x Vertical Color-Flip + Exact Architectural Parity")
    print(f"  Batch Size:              {args.batch_size:,} | LR: {args.lr:.2e} | Epochs: {args.epochs} + {args.finetune_eval_epochs} FT")
    print(f"  File xuất nhị phân:      {args.output}")
    print("=" * 76, flush=True)

    # 1. Đồng bộ dữ liệu từ Hugging Face Bucket nếu có cờ --sync-hf
    if args.sync_hf:
        sync_from_huggingface(args.hf_bucket, args.data)

    # 2. Nạp hoặc xây dựng cache nhị phân HKV3
    us_data, them_data, eval_data, tau_data, rho_data, rs_data = load_or_build_hkv3_cache(
        args.data, cache_path=args.cache if args.cache else None, use_cache=True
    )

    n_total = len(eval_data)
    val_count = min(100000, max(1000, int(n_total * args.val_split)))
    perm = np.random.permutation(n_total)
    val_indices = perm[:val_count]
    train_indices = perm[val_count:]

    print(f"[Split] Train: {len(train_indices):,} base ({len(train_indices) * args.symmetry:,} effective) | "
          f"Val: {len(val_indices):,} positions.", flush=True)

    train_loader = FastHKV3BatchLoader(
        us_data, them_data, eval_data, tau_data, rho_data, rs_data,
        train_indices, batch_size=args.batch_size, shuffle=True,
        symmetry=args.symmetry, feature_dropout=args.feature_dropout, device=device
    )
    val_loader = FastHKV3BatchLoader(
        us_data, them_data, eval_data, tau_data, rho_data, rs_data,
        val_indices, batch_size=args.batch_size, shuffle=False,
        symmetry=1, feature_dropout=0.0, device=device
    )

    # 3. Khởi tạo mạng HalfKPv3 với Tensor-Parallel tự nhiên trên 2 GPU (ví dụ Kaggle T4x2)
    dev0 = torch.device("cuda:0") if num_gpus >= 1 else torch.device("cpu")
    dev1 = torch.device("cuda:1") if num_gpus >= 2 else dev0
    if num_gpus >= 2:
        print(f"[T4x2 Tensor-Parallel] Nhánh Left (1024) -> {dev0} || Nhánh Right (1024) -> {dev1} (Zero PCIe Weight Copy!)", flush=True)

    base_model = HalfKPv3Network(qat=not args.no_qat, dev0=dev0, dev1=dev1)
    if args.resume and Path(args.resume).exists():
        print(f"[Resume] Đang nạp trọng số từ checkpoint {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        base_model.load_state_dict(state, strict=False)

    model: nn.Module = base_model
    criterion = HKV3MultiTaskLoss(wdl_weight=args.wdl_weight).to(dev0)
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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

    best_val_loss = float("inf")
    best_val_mae = float("inf")

    # ─── GIAI ĐOẠN 1: HUẤN LUYỆN ĐA NHIỆM ĐỒNG THỜI (EVAL + TAU + RHO + RS) ───
    print(f"\n[Stage 1] Bắt đầu huấn luyện Đa nhiệm HalfKPv3 ({args.epochs} epochs)...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_totals = {"total": 0.0, "eval": 0.0, "tau": 0.0, "rho": 0.0, "rs": 0.0, "mae_cp": 0.0}
        steps = 0

        for us_b, them_b, targets_b in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs_b = model(us_b, them_b)
                loss, m = criterion(outputs_b, targets_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            base_model.clamp_quantizable_weights()

            for k in train_totals:
                train_totals[k] += m[k]
            steps += 1

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

        ckpt_path = Path(args.output).with_suffix(f".epoch{epoch}.pt")
        torch.save({"epoch": epoch, "model_state_dict": base_model.state_dict()}, ckpt_path)

        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            best_val_mae = val_m["mae_cp"]
            export_hkv3_binary(base_model, args.output)

    # ─── GIAI ĐOẠN 2: TINH CHỈNH CHUYÊN SÂU ĐẦU ĐỌC EVAL (2A: JOINT REFINEMENT + 2B: LOCKED HEAD) ───
    if args.finetune_eval_epochs > 0:
        phase_a = max(1, args.finetune_eval_epochs // 2)
        phase_b = max(1, args.finetune_eval_epochs - phase_a)
        print(f"\n[Stage 2] Tinh chỉnh chuyên sâu Eval ({phase_a} epochs Phase 2A + {phase_b} epochs Phase 2B)...", flush=True)

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
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs_b = model(us_b, them_b)
                    loss_ev = criterion.compute_eval_loss(outputs_b["eval"], targets_b["eval"])
                    loss_aux = (
                        F.binary_cross_entropy_with_logits(outputs_b["tau"], targets_b["tau"])
                        + F.binary_cross_entropy_with_logits(outputs_b["rho"], targets_b["rho"])
                        + F.binary_cross_entropy_with_logits(outputs_b["rs"], targets_b["rs"])
                    )
                    loss = 4.0 * loss_ev + 0.15 * loss_aux
                scaler.scale(loss).backward()
                scaler.unscale_(ft_opt_a)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=0.5)
                scaler.step(ft_opt_a)
                scaler.update()
                base_model.clamp_quantizable_weights()

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
                export_hkv3_binary(base_model, args.output)

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
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs_b = model(us_b, them_b)
                    loss_ev = criterion.compute_eval_loss(outputs_b["eval"], targets_b["eval"])
                scaler.scale(loss_ev).backward()
                scaler.unscale_(ft_opt_b)
                scaler.step(ft_opt_b)
                scaler.update()
                base_model.clamp_quantizable_weights()

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
                export_hkv3_binary(base_model, args.output)

        for p in base_model.parameters():
            p.requires_grad = True

    # ─── KIỂM CHỨNG ĐỐI XỨNG SỐ NGUYÊN C++ AVX2 SAU KHI XUẤT FILE ───
    quant_tensors = export_hkv3_binary(base_model, args.output)
    sample_n = min(64, len(val_indices))
    verify_integer_avx2_parity_and_accuracy(
        quant_tensors,
        us_data[val_indices[:sample_n]],
        them_data[val_indices[:sample_n]],
    )
    print(f"\n[Done] Huấn luyện hoàn tất! Best Validation MAE: {best_val_mae:.2f} cp -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
