import json
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def create_prompt(problem):
    """Create a prompt for the problem."""
    return problem + " Your answer should be wrapped by the \\boxed{} tag."
def transform_dataset_to_jsonl(dataset_name, output_dir, limit=None, num_splits=1):
    """Transform the dataset to JSONL format."""
    print(f"Loading {dataset_name} dataset from Hugging Face...")
    dataset = load_dataset(dataset_name, trust_remote_code=True, split="train")
    
    if limit:
        # Limit to the specified number of problems
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Limited dataset to first {len(dataset)} problems (out of {len(dataset)} total)")
    else:
        print(f"Processing all {len(dataset)} problems")
    
    # Create output directory if it doesn't exist
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate items per split
    total_items = len(dataset)
    items_per_split = total_items // num_splits
    remainder = total_items % num_splits
    
    # Transform and write to JSONL files
    count = 0
    for split_idx in range(num_splits):
        # Calculate start and end indices for this split
        start_idx = split_idx * items_per_split + min(split_idx, remainder)
        end_idx = (split_idx + 1) * items_per_split + min(split_idx + 1, remainder)
        
        # Create split filename
        split_file = f"{output_dir}/split_{split_idx}.jsonl"
        
        split_count = 0
        with open(split_file, 'w') as f:
            for i in tqdm(range(start_idx, end_idx), desc=f"Split {split_idx+1}/{num_splits}"):
                example = dataset[i]
                problem = example["problem"]
                answer = example["answer"]
                
                # Create the prompt
                prompt = create_prompt(problem)
                
                # Create the JSON object
                json_obj = {
                    "custom_id": f"problem-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.6,
                        "max_tokens": 16384
                    }
                }
                
                # Write to file
                f.write(json.dumps(json_obj) + '\n')
                split_count += 1
                count += 1
        
        print(f"Split {split_idx+1}: Wrote {split_count} items to {split_file}")
    
    print(f"Successfully transformed {count} problems to JSONL format")
    if num_splits > 1:
        print(f"Output split into {num_splits} files")
    else:
        print(f"Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Transform DeepScaler dataset to JSONL format")
    parser.add_argument("--dataset", default="agentica-org/DeepScaleR-Preview-Dataset", 
                        help="Dataset name on Hugging Face")
    parser.add_argument("--output", default="deepscaler_problems", 
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit to the first N problems (default: all)")
    parser.add_argument("--splits", type=int, default=1,
                        help="Number of files to split the output into (default: 1)")
    
    args = parser.parse_args()
    
    transform_dataset_to_jsonl(args.dataset, args.output, args.limit, args.splits)

if __name__ == "__main__":
    main()