from huggingface_hub import hf_hub_download, list_repo_files
import os

REPO_ID = "LLM360/guru_RL"
REPO_TYPE = "dataset"

all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)

# split名到本地目录的映射
split_to_local = {
    "train": "training_data",
    "online_eval": "online_eval_data",
    "offline_eval": "offline_eval_data"
}

def download_files_from_split(split, local_dir):
    if split == "train":
        parquet_files = [f for f in all_files if "/" not in f and f.endswith(".parquet")]
    else:
        parquet_files = [f for f in all_files if f.startswith(f"{split}/") and f.endswith(".parquet")]
    os.makedirs(local_dir, exist_ok=True)
    for filename in parquet_files:
        print(f"Downloading {filename} to {local_dir} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False        )

for split, local_dir in split_to_local.items():
    download_files_from_split(split, local_dir)