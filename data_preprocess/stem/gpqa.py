"""
Preprocess the GPQA dataset to parquet format
"""

import os
import argparse
import random
from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset


def get_datasets():
    """
    Loads the GPQA dataset.
    """
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
        print(f"GPQA dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None



def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):
        # Create a default "skip" response with all required fields
        question = example["Question"].strip()
        correct = example["Correct Answer"].strip()
        incorrect1 = example["Incorrect Answer 1"].strip()
        incorrect2 = example["Incorrect Answer 2"].strip()
        incorrect3 = example["Incorrect Answer 3"].strip()

        all_choices = [correct, incorrect1, incorrect2, incorrect3]
        random.shuffle(all_choices)

        correct_index = all_choices.index(correct)
        correct_letter = chr(65 + correct_index)

        formatted_choices = ""
        for i, choice in enumerate(all_choices):
            letter = chr(65 + i)
            formatted_choices += f"({letter}) {choice}\n"
        
        # prompt format is adopted from "Zero-shot CoT Prompt" in https://www.vals.ai/benchmarks/gpqa-04-18-2025
        prompt = (
            f"What is the correct answer to this question:\n\n{question}\n\n"
            f"Choices:\n{formatted_choices}\n"
            "Reason through your answer step-by-step. Then, based on your reasoning, provide the single most likely answer choice. Answer in the format \"The correct answer is (insert answer here).\""
        )

        
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": prompt}
            ],
            "ability": "stem",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": correct_letter,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "original_prompt": prompt,
                "dataset": "Idavidrein/gpqa",
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            print(f'\none prompt example is \n{prompt}')
            
        return data

    return process_fn

if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save GPQA dataset.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="stem",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="gpqa",
                        help='Name of the dataset.')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Number of samples to use from dataset. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    test_output_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Load the dataset
    _, dataset = get_datasets()

    # Process the dataset
    process_fn = make_map_fn('test', data_source)
    
    dataset = dataset.map(function=process_fn, with_indices=True)

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