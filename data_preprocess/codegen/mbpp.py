"""Downloads, processes, and saves MBPP datasets."""

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
    Loads the MBPP dataset.
    """
    try:
        dataset = load_dataset("google-research-datasets/mbpp", cache_dir=cache_dir)
        print(f"Train set: {len(dataset['train'])} examples")
        print(f"Test set: {len(dataset['test'])} examples")
        return dataset["train"], dataset["test"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def make_map_fn(split: str, data_source: str, prompt_style: str="zero_style") -> callable:
    def process_fn(example, idx):
        # Create prompt
        prompt = (
            f"{example['text']}\n\n"
            f"Your solution should be a complete, self-contained function in a markdown code block. "
            f"Make sure your solution passes the following test cases:\n"
        )
        
        # Construct test code
        test_code = ""
        if example.get('test_setup_code'):
            test_code += example['test_setup_code'] + "\n\n"
        
        # Add all test assertions
        for assertion in example['test_list'] + example.get('challenge_test_list', []):
            test_code += assertion + "\n"
        
        # Validate that the canonical solution passes the tests
        solution = example['code']
        full_code = f"{solution}\n\n{test_code}"
        
        succ, err = code_exec(full_code)
        if not succ:
            print(f"Test code failed for example {idx}: {example.get('task_id', 'unknown')}")
            return {
                "data_source": None,
                "prompt": None,
                "raw_prompt": None,
                "ability": None,
                "reward_model": None,
                "extra_info": None
            }
        
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
                    "functional": test_code
                }),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": solution,
                "prompt": prompt,
                "dataset": "mbpp",
                "task_id": str(example.get("task_id", "")),
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save MBPP datasets.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="codegen",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="mbpp",
                        help='Name of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=None,
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--test-sample-size', type=int, default=None,
                        help='Number of samples to use from test dataset. If None, use all samples.')
    parser.add_argument('--prompt-style', type=str, choices=['zero_style'], default='zero_style',
                        help='Prompt style to use (currently only zero_style supported).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}_{args.name}"
    train_output_dir = os.path.join(args.data_dir, 'train')
    test_output_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Load the dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    train_dataset, test_dataset = get_datasets(cache_dir=cache_dir)

    # Process the dataset
    process_train_fn = make_map_fn('train', data_source, args.prompt_style)
    process_test_fn = make_map_fn('test', data_source, args.prompt_style)
    
    train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)
    test_dataset = test_dataset.map(function=process_test_fn, with_indices=True)

    # Filter out examples where processing failed
    train_dataset = train_dataset.filter(lambda x: x["data_source"] == data_source)
    test_dataset = test_dataset.filter(lambda x: x["data_source"] == data_source)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        train_dataset = train_dataset.filter(lambda x: length_filter.check(x))
        test_dataset = test_dataset.filter(lambda x: length_filter.check(x))
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Sample the datasets using utility function
    train_dataset = sample_dataset(train_dataset, args.train_sample_size)
    test_dataset = sample_dataset(test_dataset, args.test_sample_size)

    # Save the datasets using utility function
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=data_source,
        sample_size=len(train_dataset)
    )
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=data_source,
        sample_size=len(test_dataset)
    )

    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Train data saved to {train_output_path} ({len(train_dataset)} samples)\n"
          f"Test data saved to {test_output_path} ({len(test_dataset)} samples)") 