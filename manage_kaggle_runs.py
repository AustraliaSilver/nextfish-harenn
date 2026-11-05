#!/usr/bin/env python3
"""
Nextfish HARENN - Kaggle 5x CPU Worker Manager
Tự động giám sát, kích hoạt và duy trì 5 phiên chạy CPU độc lập trên Kaggle (4 vCPU, 30GB RAM, Hash 4096MB)
để cùng đồng hành với cụm GitHub Actions sinh dữ liệu lên Hugging Face.
"""

import os
import sys
import json
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_USERNAME = "hunganh12333"

def get_kaggle_worker_code(worker_num: int, hf_token: str) -> str:
    return f'''#!/usr/bin/env python3
import os, sys, subprocess, time, glob, json
from datetime import datetime

print("====================================================================")
print("  NEXTFISH HARENN KAGGLE DATAGEN WORKER {worker_num} (4 vCPUs | 30GB RAM)")
print("====================================================================")

# 1. Download Stockfish 19 Universal Linux
if not os.path.exists("./harenn_engine"):
    print("[1/4] Downloading Stockfish 19 Universal Linux...")
    cmd = "curl -L -o stockfish.tar.gz https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz && tar -xzvf stockfish.tar.gz && mv stockfish/stockfish-linux-x86-64-universal ./harenn_engine && chmod +x harenn_engine"
    subprocess.check_call(cmd, shell=True)

# 2. Download latest parallel_generate.py from branch data
print("[2/4] Downloading latest parallel_generate.py...")
cmd = "curl -L -o parallel_generate.py https://raw.githubusercontent.com/AustraliaSilver/nextfish-harenn/data/parallel_generate.py"
subprocess.check_call(cmd, shell=True)

# 3. Install dependencies
print("[3/4] Installing python-chess & huggingface_hub...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "python-chess", "huggingface_hub"])

# 4. Run data generation (11.5 hours budget, 4096MB Hash/proc = 16GB total on 30GB RAM, 16384 nodes, 4 vCPUs)
print("[4/4] Starting Turbo Parallel Generator...")
os.makedirs("data", exist_ok=True)
gen_cmd = [
    sys.executable, "parallel_generate.py",
    "--engine", "./harenn_engine",
    "--output", "./data",
    "--games", "10000",
    "--batch-size", "100",
    "--concurrency", "4",
    "--hash", "4096",
    "--time-budget", "41400",
    "--worker-id", "k{worker_num}",
    "--nodes", "16384"
]
subprocess.run(gen_cmd)

# 5. Pack and Upload to Hugging Face
print("=== Packing data files into single JSONL ===")
from huggingface_hub import HfApi

hf_token = "{hf_token}"
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
    except Exception:
        pass
files = glob.glob("data/*.json") + glob.glob("data/*.jsonl")
print(f"Total raw files generated: {{len(files)}}")
if files and hf_token:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    packed_filename = f"hnn_kaggle_k{worker_num}_{{ts}}.jsonl"
    packed_path = os.path.join("data", packed_filename)
    total_positions = 0
    with open(packed_path, "w", encoding="utf-8") as out_f:
        for f in files:
            if f == packed_path:
                continue
            if f.endswith(".json"):
                try:
                    with open(f, "r", encoding="utf-8") as in_f:
                        d = json.load(in_f)
                        for pos in d.get("positions", []):
                            out_f.write(json.dumps(pos) + "\\n")
                            total_positions += 1
                except Exception:
                    pass
            elif f.endswith(".jsonl"):
                try:
                    with open(f, "r", encoding="utf-8") as in_f:
                        for line in in_f:
                            line = line.strip()
                            if line:
                                out_f.write(line + "\\n")
                                total_positions += 1
                except Exception:
                    pass

    size_mb = os.path.getsize(packed_path) / (1024 * 1024)
    print(f"Packed {{total_positions:,}} positions into {{packed_filename}} ({{size_mb:.2f}} MB)")
    api = HfApi(token=hf_token)
    bucket_id = "australia1/myrandombucket"
    print(f"Uploading {{packed_filename}} to Hugging Face {{bucket_id}}...")
    api.batch_bucket_files(bucket_id, add=[(packed_path, packed_filename)])
    print("Upload completed successfully!")
'''

def manage_5_kaggle_workers():
    print("=== Checking & Managing 5 Kaggle CPU Workers ===")
    force = "--force" in sys.argv
    if force:
        print("  [Note] --force flag detected: Restarting all 5 workers with latest code")
    token = os.environ.get("KAGGLE_API_TOKEN") or "KGAT_b42d4a8ff82836dd870f04137475c469"
    os.environ["KAGGLE_API_TOKEN"] = token

    api = KaggleApi()
    api.authenticate()

    hf_token = os.environ.get("HF_TOKEN", "")

    for worker_num in range(1, 6):
        kernel_slug = f"nextfish-harenn-datagen-k{worker_num}"
        kernel_ref = f"{KAGGLE_USERNAME}/{kernel_slug}"
        print(f"\n[Worker {worker_num}] Checking {kernel_ref}...")

        is_running = False
        if not force:
            try:
                status_obj = api.kernels_status(kernel_ref)
                cur_status = status_obj.get("status", "").upper() if isinstance(status_obj, dict) else str(status_obj).upper()
                print(f"  Current status: {cur_status}")
                if "RUNNING" in cur_status or "QUEUED" in cur_status:
                    print(f"  Worker {worker_num} is actively running ({cur_status}). No action needed.")
                    is_running = True
            except Exception as e:
                print(f"  Status check: {e} (Will create/push fresh kernel)")

        if not is_running:
            print(f"  Pushing fresh Kaggle kernel for Worker {worker_num}...")
            tmpdir = tempfile.mkdtemp()
            meta = {
                "id": kernel_ref,
                "title": f"Nextfish HARENN Datagen K{worker_num}",
                "code_file": "worker.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": True
            }
            with open(os.path.join(tmpdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            worker_code = get_kaggle_worker_code(worker_num, hf_token)
            with open(os.path.join(tmpdir, "worker.py"), "w", encoding="utf-8") as f:
                f.write(worker_code)

            try:
                res = api.kernels_push(tmpdir)
                ref_url = res.get("url") if isinstance(res, dict) else kernel_ref
                print(f"  Successfully pushed Worker {worker_num} -> {ref_url}")
            except Exception as e:
                print(f"  [Error] Failed to push Worker {worker_num}: {e}")

    print("\n=== Kaggle 5x Manager execution complete! ===")

if __name__ == "__main__":
    manage_5_kaggle_workers()
