# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess a puzzle dataset to parquet format.
"""
import os
import datasets
import argparse
import re
from sklearn.model_selection import train_test_split

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
)

from verl.utils.data_process.prompt import build_zero_style_prompt
from verl.utils.data_process.utils import add_suffix, set_seed, sample_dataset, save_dataset

# Constants
INSTRUCTION_FOLLOW = "Please put your answer within <answer> and </answer> tags, for example <answer> ['pigeon', 'sparrow', 'quail'] </answer>."

def make_prefix(dp, model_type = 'instruct'):
    """
    Create a suitable prompt prefix based on model type.
    
    Args:
        dp (dict): Data point containing input, instruction, etc.
        model_type (str): 'instruct' or 'base'
        
    Returns:
        str: Formatted prompt
    """
    constraints = dp['input']
    result = dp['ground_truth']
    instruction = dp['instruction']
    if model_type == 'instruct':
        prefix = f"{instruction}. The constraints are: {constraints}. Think step by step to find the answer. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> ['pigeon', 'sparrow', 'quail'] </answer>."
    else:
        prefix = f"{instruction}. The constraints are: {constraints}."
    return prefix


def extract_from_ground_truth(text):
    """
    Extract answer from the ground truth text.
    
    Args:
        text: Ground truth answer, could be string or list
        
    Returns:
        list: Extracted answer
    """
    if isinstance(text, list):
        return text
    else:
        return eval(text)


def make_map_fn(split: str, data_source: str, model_type: str) -> callable:
    """
    Create a mapping function for processing dataset examples.
    
    Args:
        split (str): Data split ('train' or 'test')
        data_source (str): Name of the data source
        model_type (str): Model type ('instruct' or 'base')
        
    Returns:
        callable: Function to map over the dataset
    """
    def process_fn(example, idx):
        # Generate the appropriate question format based on model type
        question = make_prefix(example, model_type)
        num_objects = example['num_objects']
        final_arrangement = extract_from_ground_truth(example['ground_truth'])
        
        if model_type == 'instruct':
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "logical_reasoning",
                "reward_model": {
                        "style": "rule",
                        "ground_truth": final_arrangement
                    },
                "apply_chat_template": True,
                "extra_info": {
                    'id': example['id'] if 'id' in example else str(idx),
                    'raw_instruction': example['instruction'],
                    'raw_input': example['input'],
                    'num_objects': num_objects,
                    'split': split,
                }
            }
        elif model_type == 'base':
            # Use the build_zero_style_prompt function for base model
            # The question variable already has the appropriate format from make_prefix
            prompt = build_zero_style_prompt(extra_instruction=INSTRUCTION_FOLLOW)
            
            data = {
                "data_source": data_source,
                "prompt": [],  # no messages-like prompt. instead, use from-scratch raw_prompt
                "raw_prompt": prompt.replace("{{prompt}}", question),
                "ability": "logical_reasoning",
                "reward_model": {
                        "style": "rule",
                        "ground_truth": final_arrangement
                    },
                "apply_chat_template": False,
                "extra_info": {
                    'id': example['id'] if 'id' in example else str(idx),
                    'raw_instruction': example['instruction'],
                    'raw_input': example['input'],
                    'num_objects': num_objects,
                    'split': split,
                }
            }
        
        if idx == 0:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data
        
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='data/puzzles_dataset/puzzles_dataset.json', help='Path to json file')
    parser.add_argument('--output_dir', default='../../../data', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.2, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.02, help='Proportion of data for test set')
    parser.add_argument('--data_source', default='ordering_puzzle_dataset', help='Name of data source')
    parser.add_argument('--model_type', default='base', choices = ['base', 'instruct'], help='Model type base or instruct')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load dataset from JSON
    dataset = datasets.load_dataset('json', data_files=args.json_path)['train']
    print(f"Loaded dataset with {len(dataset)} examples")
   
    # Validate train and test sizes
    if args.train_size + args.test_size > 1.0:
        raise ValueError(f"The sum of train_size ({args.train_size}) and test_size ({args.test_size}) cannot exceed 1.0")
   
    # Transform dataset
    process_train_fn = make_map_fn('train', args.data_source, args.model_type)
    processed_dataset = dataset.map(function=process_train_fn, with_indices=True)
    
    # Split dataset into train and test
    train_indices, test_indices = train_test_split(
        range(len(dataset)), 
        train_size=args.train_size,
        test_size=args.test_size, 
        random_state=args.seed
    )
    
    print(f"Train set size: {len(train_indices)}, Test set size: {len(test_indices)}")
  
    # Create train and test datasets
    train_dataset = processed_dataset.select(train_indices)
    test_dataset = processed_dataset.select(test_indices)
    
    # Store the original training dataset size
    original_train_size = len(train_dataset)
    
    # Sample the training dataset if needed
    train_dataset = sample_dataset(train_dataset, args.train_sample_size)
    
    # Create output directories
    train_output_dir = os.path.join(args.output_dir, "train")
    test_output_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Save train dataset
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=f"logic__{args.data_source}",
        sample_size=args.train_sample_size if args.train_sample_size else len(train_dataset)
    )
    
    # Save test dataset
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=f"logic__{args.data_source}",
        sample_size=len(test_dataset)
    )
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=args.output_dir, dst=args.hdfs_dir)
            print(f"Data copied to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("HDFS utilities not available. Install verl package for HDFS support.")
            
    print(f"Done! \n"
          f"Train data saved to {train_output_path}\n"
          f"Test data saved to {test_output_path}")
    print(f"Original train set size: {original_train_size} examples")
    print(f"Final train set size: {len(train_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")