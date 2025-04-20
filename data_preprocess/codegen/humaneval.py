"""Downloads, processes, and saves HumanEval dataset."""

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset

from verl.utils.data_process.prompt import build_zero_style_prompt
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter
from verl.utils.reward_score.coder1 import code_exec


def get_datasets(cache_dir: str):
    """
    Loads the HumanEval dataset.
    """
    try:
        dataset = load_dataset("openai_humaneval", cache_dir=cache_dir)["test"]
        print(f"HumanEval dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def make_map_fn(split: str, data_source: str, prompt_style: str="zero_style") -> callable:
    def process_fn(example, idx):
        # Create a default "skip" response with all required fields
        skip_response = {
            "data_source": None,
            "prompt": None,
            "raw_prompt": None,
            "ability": None,
            "reward_model": None,
            "extra_info": None
        }
        
        # Extract task ID and prompt
        task_id = example["task_id"]
        prompt = example["prompt"]
        
        # Extract test function, entry point, and canonical solution
        test_code = example["test"]
        entry_point = example["entry_point"]
        solution = example["canonical_solution"]
        
        # Build test code that calls the entry point
        test_code_with_check = f"{test_code}\n\ncheck({entry_point})"
        
        # Verify the canonical solution passes the tests
        full_code = f"{prompt}\n{solution}\n{test_code}\n\ncheck({entry_point})"
        succ, err = code_exec(full_code, timeout=5)
        if not succ:
            print(f"Error in canonical solution for task {task_id}: {err}")
            return skip_response

        # Format the prompt according to the specified style
        raw_prompt = build_zero_style_prompt(prompt=prompt)
        data = {
            "data_source": data_source,
            "prompt": [],
            "raw_prompt": raw_prompt,
            "ability": "codegen",
            "apply_chat_template": False,
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    "functional": test_code_with_check
                }),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": solution,  # Include the canonical solution as reference
                "prompt": prompt,
                "dataset": "openai_humaneval",
                "task_id": task_id,
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save HumanEval dataset.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="codegen",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="humaneval",
                        help='Name of the dataset.')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples to use from dataset. If None, use all samples.')
    parser.add_argument('--prompt-style', type=str, choices=['zero_style'], default='zero_style',
                        help='Prompt style to use (currently only zero_style supported).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    test_output_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Load the dataset
    _, dataset = get_datasets(cache_dir=args.data_dir)

    # Process the dataset
    process_fn = make_map_fn('test', data_source, args.prompt_style)
    
    dataset = dataset.map(function=process_fn, with_indices=True)

    # Filter out examples where processing failed
    dataset = dataset.filter(lambda x: x["data_source"] == data_source)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        dataset = dataset.filter(lambda x: length_filter.check(x))
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Sample the dataset
    dataset = sample_dataset(dataset, args.sample_size)
    
    # Save the dataset to test directory
    test_output_path = save_dataset(
        dataset=dataset,
        output_dir=test_output_dir,
        filename_prefix=data_source,
        sample_size=len(dataset)
    )

    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Test data saved to {test_output_path} ({len(dataset)} samples)")