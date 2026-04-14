import subprocess
import os

engine_path = "../nextfish.exe"
base_output = "./data_batch_"
start_batch = 44
end_batch = 300  # To reach ~2.5k positions first

for i in range(start_batch, end_batch + 1):
    output_dir = f"{base_output}{i}"
    cmd = [
        "python",
        "simple_generate.py",
        "--engine",
        engine_path,
        "--games",
        "10",
        "--output",
        output_dir,
    ]
    print(f"Running batch {i}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in batch {i}: {result.stderr}")
        break
    else:
        print(f"Batch {i} done")

print("All batches completed")
