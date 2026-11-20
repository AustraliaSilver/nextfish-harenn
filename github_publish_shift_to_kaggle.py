#!/usr/bin/env python3
"""
GitHub End-of-Shift Kaggle Publisher:
  Chạy sau khi 17 workers của 1 ca hoàn thành.
  1. Đọc danh sách staging/r{run_id}/*.meta.json từ HF Dataset (australia1/harenn-hkv3-npz).
  2. Tải các shard .npz chưa có trong packed_manifest.json.
  3. Gom theo từng cụm (<= 5.5 GB / part) đăng lên Kaggle Dataset pXXX và cập nhật packed_manifest.json.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from kaggle_pack_npz_dataset import (
    get_secret,
    init_kaggle_api,
    load_manifest,
    publish_shards_to_kaggle,
    save_and_push_manifest,
)


def main():
    parser = argparse.ArgumentParser(description="Publish GitHub run NPZ shards from HF NPZ Bucket/Dataset to Kaggle")
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument("--hf-npz-bucket", default="australia1/harenn-hkv3-npz-bucket", type=str)
    parser.add_argument("--hf-dataset", default="", type=str)
    parser.add_argument("--hf-bucket", default="australia1/myrandombucket", type=str)
    parser.add_argument("--kaggle-owner", default="", type=str)
    parser.add_argument("--dataset-prefix", default="harenn-hkv3-npz", type=str)
    parser.add_argument("--index-dataset", default="harenn-hkv3-npz-index", type=str)
    parser.add_argument("--max-part-gb", default=5.5, type=float)
    parser.add_argument("--public", action="store_true", help="(Tuong thich nguoc) Mac dinh da la public")
    parser.add_argument("--private", action="store_true", help="Tao Kaggle Dataset dang private (mac dinh public)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cleanup-hf-npz", action="store_true", help="Xoa NPZ + metadata cua run-id tren HF NPZ staging sau khi publish Kaggle xong (giu nguyen HF Bucket JSONL)")
    parser.add_argument("--cleanup-hf-test", action="store_true", help="Xoa du lieu test cua run-id tren ca HF NPZ staging & HF Bucket (chi dung cho smoke test)")
    parser.add_argument("--ignore-running-packer", action="store_true", help="Bo qua kiem tra kernel packer dang chay")
    args = parser.parse_args()

    hf_token = get_secret("HF_TOKEN")
    if not hf_token:
        print("::error::HF_TOKEN secret is not set!")
        sys.exit(1)

    from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download

    hf_api = HfApi(token=hf_token)
    work_dir = Path(f"/tmp/kaggle_pub_{args.run_id}")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    staging_prefix = f"staging/r{args.run_id}/"
    use_npz_bucket = bool(args.hf_npz_bucket.strip())

    def fetch_hf_file(repo_path: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_p = dest_dir / Path(repo_path).name
        if use_npz_bucket:
            hf_api.download_bucket_files(
                args.hf_npz_bucket,
                files=[(repo_path, str(local_p))],
                raise_on_missing_files=True,
            )
            return local_p
        lp = Path(hf_hub_download(
            repo_id=args.hf_dataset,
            repo_type="dataset",
            filename=repo_path,
            token=hf_token,
            local_dir=str(dest_dir),
        ))
        if lp != local_p:
            shutil.move(lp, local_p)
        return local_p

    try:
        if use_npz_bucket:
            tree = hf_api.list_bucket_tree(args.hf_npz_bucket, prefix=staging_prefix, recursive=True)
            repo_files = [item.path for item in tree if getattr(item, "path", "").endswith(".meta.json")]
        else:
            repo_files = hf_api.list_repo_files(repo_id=args.hf_dataset, repo_type="dataset")
    except Exception as e:
        src_name = args.hf_npz_bucket if use_npz_bucket else args.hf_dataset
        print(f"[Info] Chua doc duoc {src_name}: {e}")
        return

    meta_paths = sorted(p for p in repo_files if p.startswith(staging_prefix) and p.endswith(".meta.json"))
    print(f"[HF] Tim thay {len(meta_paths)} shard metadata cho run {args.run_id}", flush=True)
    if not meta_paths:
        print("[Done] Khong co shard nao de dang len Kaggle.")
        return

    metas: List[Dict] = []
    for mp in meta_paths:
        lp = fetch_hf_file(mp, work_dir / "meta")
        metas.append(json.loads(lp.read_text(encoding="utf-8")))

    k_api, detected_user = init_kaggle_api()
    owner = args.kaggle_owner or detected_user or "hunganh12333"
    if not args.dry_run and not args.ignore_running_packer:
        try:
            st = str(k_api.kernels_status(f"{owner}/harenn-hf-to-npz-packer").status).upper()
            if "RUNNING" in st or "QUEUED" in st:
                print(
                    f"[Skip Kaggle] {owner}/harenn-hf-to-npz-packer dang {st} -> tam bo qua publish Kaggle de tranh xung dot part/manifest (NPZ va JSONL da co tren HF).",
                    flush=True,
                )
                return
        except Exception as e:
            print(f"[Warn] Khong kiem tra duoc status packer ({e}), van tiep tuc.", flush=True)

    index_ref = f"{owner}/{args.index_dataset}" if "/" not in args.index_dataset else args.index_dataset
    manifest = load_manifest(k_api, index_ref, work_dir)
    done_set = set(manifest.get("processed_files", []))

    pending = [m for m in metas if m.get("jsonl") not in done_set]
    print(f"[Manifest] Da co {len(done_set)} file | Chua len Kaggle trong ca nay: {len(pending)}", flush=True)

    if pending and not args.dry_run:
        max_part_bytes = int(max(1.0, args.max_part_gb) * 1024 * 1024 * 1024)
        groups: List[List[Dict]] = []
        cur_grp: List[Dict] = []
        cur_bytes = 0
        for m in pending:
            sz = int(m.get("npz_bytes", 0))
            if cur_grp and (cur_bytes + sz > max_part_bytes or len(cur_grp) >= 17):
                groups.append(cur_grp)
                cur_grp = []
                cur_bytes = 0
            cur_grp.append(m)
            cur_bytes += sz
        if cur_grp:
            groups.append(cur_grp)

        part_num = int(manifest.get("next_part", 1))
        for g_idx, grp in enumerate(groups, 1):
            dl_dir = work_dir / f"grp_{g_idx}"
            dl_dir.mkdir(parents=True, exist_ok=True)
            shard_paths: List[Path] = [fetch_hf_file(m["shard_repo_path"], dl_dir) for m in grp]

            ds_ref = publish_shards_to_kaggle(k_api, owner, args.dataset_prefix, part_num, shard_paths, not args.private)
            src_files = [m["jsonl"] for m in grp]
            raw_sum = sum(int(m["raw_positions"]) for m in grp)
            uniq_sum = sum(int(m["unique_positions"]) for m in grp)
            manifest["processed_files"].extend(src_files)
            manifest["parts"].append(
                {
                    "part": part_num,
                    "dataset": ds_ref,
                    "files": len(src_files),
                    "shards": len(shard_paths),
                    "raw_positions": raw_sum,
                    "unique_positions": uniq_sum,
                }
            )
            part_num += 1
            manifest["next_part"] = part_num
            save_and_push_manifest(k_api, index_ref, manifest, work_dir)
            shutil.rmtree(dl_dir, ignore_errors=True)
    elif args.dry_run:
        # Tai thu 1 shard de kiem tra tron ven luong tai tu HF
        sp = fetch_hf_file(metas[0]["shard_repo_path"], work_dir / "dry_check")
        print(f"[Dry-Run] Da tai thanh cong {sp} ({sp.stat().st_size:,} bytes). Bo qua tao Kaggle Dataset.", flush=True)

    if args.cleanup_hf_npz or args.cleanup_hf_test:
        del_paths = list(meta_paths) + [m["shard_repo_path"] for m in metas]
        if use_npz_bucket:
            hf_api.batch_bucket_files(args.hf_npz_bucket, delete=del_paths)
            print(f"[Cleanup] Da xoa {len(del_paths)} file NPZ/meta tren private bucket '{args.hf_npz_bucket}'.", flush=True)
        else:
            del_ops = [CommitOperationDelete(path_in_repo=p) for p in del_paths]
            hf_api.create_commit(
                repo_id=args.hf_dataset,
                repo_type="dataset",
                operations=del_ops,
                commit_message=f"Cleanup NPZ staging run {args.run_id}",
            )
            print(f"[Cleanup] Da xoa {len(del_paths)} file NPZ/meta tren HF dataset '{args.hf_dataset}'.", flush=True)

    if args.cleanup_hf_test:
        test_jsonl = [m["jsonl"] for m in metas if m.get("jsonl")]
        if test_jsonl:
            hf_api.batch_bucket_files(args.hf_bucket, delete=test_jsonl)
        print(f"[Cleanup] Da xoa {len(test_jsonl)} file test tren HF Bucket '{args.hf_bucket}'.", flush=True)


if __name__ == "__main__":
    main()
