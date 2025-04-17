import os
import json
import sys
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from rich.rule import Rule

import rich
import matplotlib.pyplot as plt
import datasets
import time
import argparse
import random
import numpy as np
import torch
import transformers
from bs4 import BeautifulSoup

from verl.utils.data_process.filter import Filter, LengthFilter

class MultiHierTTFilter(LengthFilter):
    def __init__(self, tokenizer, max_length=2048):
        super().__init__(tokenizer=tokenizer, max_length=max_length)

    def check(self, example):
        print(example["qa"])
        # Only keep arithmetic questions
        question_type = example["qa"]["question_type"]
        if question_type != "arithmetic":
            return False
        # Filter out questions that need text evidence
        text_evidence = example["qa"]["text_evidence"]
        if text_evidence != []:
            return False
        # Ensure the prompt length is within the specified range
        length_check = super().check(example)
        # length_check = (self.min_length <= len(example["raw_prompt"].split()) <= self.max_length)
        if not length_check:
            return False
        
        return True


WORKDING_DIR = os.path.join(os.environ.get("HOME"), "Reasoning360")

Prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here.

User: You are given one or more tables. Use the information in the tables to answer the following question.
{{tables}}
The question is:
{{question}}
Please put your answer in \\boxed{} tags.
Assistant: <think>
"""


def get_dataset(cache_dir, download=False):
    data_path = os.path.join(cache_dir, "multihier/")
    if download: 
        if os.path.exists(data_path):
            pass
        else:
            import gdown

            # Replace with your folder ID or full URL
            url = "https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ"

            # Download the folder to the cache directory
            gdown.download_folder(url, output=data_path, quiet=False, use_cookies=False)
    
    train_path = os.path.join(data_path, "MultiHiertt_dataset/train.json")
    test_path = os.path.join(data_path, "MultiHiertt_dataset/dev.json")
    
    def _align_format(dataset):
        for entry in dataset:
            entry["qa"]["answer"] = str(entry["qa"]["answer"])
        return dataset
    
    with open(train_path, "r") as f:
        train_dataset = json.load(f)
        train_dataset = _align_format(train_dataset)
    with open(test_path, "r") as f:
        test_dataset = json.load(f)
        test_dataset = _align_format(test_dataset)
    print("Total train samples:", len(train_dataset))
    print("Total test samples:", len(test_dataset))
    
    return Dataset.from_list(train_dataset), Dataset.from_list(test_dataset)


def html_table_to_markdown(table):
    """
    Convert HTML table to markdown format.
    Handles hierarchical tables with colspan attributes.
    
    Args:
        table (str): HTML table string
    
    Returns:
        str: Markdown representation of the table
    """
    import re
    from bs4 import BeautifulSoup
    
    # Parse HTML
    soup = BeautifulSoup(table, 'html.parser')
    
    # Get all rows
    rows = soup.find_all('tr')
    if not rows:
        return ""
    
    # Process each row to determine column structure
    table_data = []
    max_cols = 0
    
    for row in rows:
        row_data = []
        col_idx = 0
        
        for cell in row.find_all(['td', 'th']):
            # Get cell content
            content = cell.get_text().strip()
            
            # Get colspan (default to 1)
            colspan = int(cell.get('colspan', 1))
            
            # Add cell with colspan info
            row_data.append({
                'content': content,
                'colspan': colspan,
                'col_idx': col_idx
            })
            
            col_idx += colspan
        
        max_cols = max(max_cols, col_idx)
        table_data.append(row_data)
    
    # Create markdown table
    markdown_rows = []
    
    # Process each row
    for i, row in enumerate(table_data):
        md_row = [''] * max_cols
        
        # Fill in cells
        for cell in row:
            content = cell['content']
            col_idx = cell['col_idx']
            colspan = cell.get('colspan', 1)
            
            # For cells with colspan > 1, center the content
            if colspan > 1:
                md_row[col_idx] = content
                # Fill in empty cells for the span
                for j in range(1, colspan):
                    if col_idx + j < max_cols:
                        md_row[col_idx + j] = ''
            else:
                md_row[col_idx] = content
        
        # Join cells with pipe separator
        markdown_rows.append('| ' + ' | '.join(md_row) + ' |')
        
        # Add header separator after first row
        if i == 0:
            separator = '| ' + ' | '.join(['---'] * max_cols) + ' |'
            markdown_rows.append(separator)
    
    return '\n'.join(markdown_rows)
    

def _tiny_analyze(dataset):
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    max_length = 0
    min_length = 100000
    mean_length = 0
    std_length = 0
    num_arithmetic = 0
    num_need_text = 0
    for entry in dataset:
        text_string = " ".join(entry["paragraphs"])
        table_string = "\n".join(entry["tables"])
        question = entry["qa"]["question"]
        answer = entry["qa"]["answer"]
        prompt = text_string + table_string + question
        prompt_tokens = tokenizer.tokenize(prompt)
        max_length = max(max_length, len(prompt_tokens))
        min_length = min(min_length, len(prompt_tokens))
        mean_length += len(prompt_tokens)
        std_length += len(prompt_tokens) ** 2
        question_type = entry["qa"]["question_type"]
        if question_type == "arithmetic":
            num_arithmetic += 1
        need_text = entry["qa"]["text_evidence"] != []
        if need_text:
            num_need_text += 1
    mean_length /= len(dataset)
    std_length = np.sqrt(std_length / len(dataset) - mean_length ** 2)
    print("=" * 10 + "Tiny Analyze" + "=" * 10)
    print(f"Max prompt length: {max_length}")
    print(f"Min prompt length: {min_length}")
    print(f"Mean prompt length: {mean_length}")
    print(f"Std prompt length: {std_length}")
    print(f"Number of arithmetic questions: {num_arithmetic}")
    print(f"Number of questions need text: {num_need_text}")

# Input Prediction
def make_map_fn(split):

    def process_fn(example, idx):
        try:
            tables = example.pop("tables")
            paragraphs = example.pop("paragraphs")
            question = example["qa"]["question"]
            answer = example["qa"]["answer"]
            table_string = "\n".join([html_table_to_markdown(table) for table in tables])
            raw_prompt = Prompt.replace("{{tables}}", table_string).replace("{{question}}", question)
        except Exception as e:
            print(e)
            print(tables)
            exit()
        data = {
            "data_source": "tablereason-multihiertt",
            "prompt": [],
            "raw_prompt": raw_prompt,
            "ability": "tablereason",
            "apply_chat_template": False,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, 
                            "index": idx,
                           },
        }
        if idx == 0 or idx == 1:
            print("=" * 10 + f"{split} {idx}" + "=" * 10)
            print(data)
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/multihier', help='Directory to save the processed data files')
    parser.add_argument('--prompt-style', type=str, choices=['zero_style'], default='zero_style',
                        help='Prompt style to use: zero_style or instruction')
    parser.add_argument('--train-sample-size', type=int, default=None, 
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--test-sample-size', type=int, default=None,
                        help='Number of samples to use from test dataset. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Modify output directory name based on train sample size
    output_dir_name = args.output_dir
    if args.train_sample_size is not None:
        if (args.train_sample_size / 1000) % 1 != 0:
            size_str = f"{args.train_sample_size / 1000:.1f}k"
        else:
            size_str = f"{args.train_sample_size // 1000}k"
            
        if "_" in output_dir_name:
            name_parts = output_dir_name.split("_", 1)
            output_dir_name = f"{name_parts[0]}{size_str}_{name_parts[1]}"
        else:
            output_dir_name = f"{output_dir_name}{size_str}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_name, exist_ok=True)

    # Get CodeIO dataset
    train_dataset, test_dataset = get_dataset(f"{os.path.expanduser('~')}/.cache/huggingface/datasets", download=True)
    
    # # tiny analyze
    # _tiny_analyze(train_dataset)
    # _tiny_analyze(test_dataset)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    # Calculate max lengths for train and test datasets
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    table_filter = MultiHierTTFilter(tokenizer=tokenizer, max_length=4096)
    train_dataset = train_dataset.filter(lambda x: table_filter.check(x))
    test_dataset = test_dataset.filter(lambda x: table_filter.check(x))
    
    if args.train_sample_size is not None:
        train_dataset = train_dataset.select(range(args.train_sample_size))
    if args.test_sample_size is not None:
        test_dataset = test_dataset.select(range(args.test_sample_size))

    # Save datasets
    train_dataset.to_parquet(os.path.join(output_dir_name, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(output_dir_name, 'test.parquet'))
    
    print(f"train data size:", len(train_dataset))
    print(f"test data size:", len(test_dataset))
    print(f"Output saved to: {output_dir_name}")
