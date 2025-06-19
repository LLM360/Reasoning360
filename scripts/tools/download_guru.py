import os
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "LLM360/guru-RL-92k"
REPO_TYPE = "dataset"
LOCAL_DATA_DIR = "./data"

all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
split_to_local = {"train": "train", "online_eval": "online_eval", "offline_eval": "offline_eval"}


def download_files_from_split(split, local_dir):
    parquet_files = [f for f in all_files if f.startswith(f"{split}/") and f.endswith(".parquet")]
    print(f"Downloading {len(parquet_files)} files to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    for filename in parquet_files:
        print(f"Downloading {filename} to {local_dir}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        # Remove .cache under each split folder
        cache_dir = os.path.join(local_dir, ".cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


for split, local_dir in split_to_local.items():
    # download_files_from_split(split, os.path.join(LOCAL_DATA_DIR, local_dir))
    download_files_from_split(split, LOCAL_DATA_DIR)
