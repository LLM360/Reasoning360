from datasets import Dataset
from huggingface_hub import create_repo, HfApi

import argparse

def upload_single_dataset(parquet_path, repo_name, organization=None, private=False):
    # Create repository
    repo_id = f"{organization}/{repo_name}" if organization else repo_name
    create_repo(repo_id, exist_ok=True)
    
    # Load dataset
    dataset = Dataset.from_parquet(parquet_path)
    
    # Push to hub
    dataset.push_to_hub(repo_id, private=private)
    print(f"Uploaded {parquet_path} to {repo_id}")
    

parser = argparse.ArgumentParser()
parser.add_argument("--parquet_path", type=str, required=True)
parser.add_argument("--repo_name", type=str, required=True)
parser.add_argument("--organization", type=str, required=False)
parser.add_argument("--private", action="store_true")
args = parser.parse_args()

# Example usage
upload_single_dataset(args.parquet_path, args.repo_name, args.organization, args.private)

# # Or upload multiple datasets to one repo
# parquet_files = ["data/train/codegen__leetcode2k_2.4k.parquet", 
#                 "data/train/codegen__taco_11.2k.parquet"]
# upload_multiple_datasets(parquet_files, "code-generation-collection")