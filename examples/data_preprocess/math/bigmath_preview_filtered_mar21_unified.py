# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset to parquet format
Code referring to https://github.com/agentica-project/deepscaler/blob/main/scripts/data/deepscaler_dataset.py
"""

import os
import datasets
from typing import Dict, List, Optional, Any, Union
import enum
import argparse
import pandas as pd
import json
import random
import numpy as np
import torch

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


class TrainDataset(enum.Enum):
    """Enum for training datasets.

    Contains identifiers for various math problem datasets used during training.
    """

    AIME = "AIME"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    OMNI_MATH = "OMNI_MATH"  # Omni Math
    NUMINA_OLYMPIAD = "OLYMPIAD"  # Unique Olympiad problems from NUMINA
    MATH = "MATH"  # Dan Hendrycks Math Problems
    STILL = "STILL"  # STILL dataset
    DEEPSCALER = "DEEPSCALER"  # DeepScaler (AIME, AMC, OMNI_MATH, MATH, STILL)


class TestDataset(enum.Enum):
    """Enum for testing/evaluation datasets.

    Contains identifiers for datasets used to evaluate model performance.
    """

    AIME = "AIME"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    MATH = "MATH"  # Math 500 problems
    MINERVA = "MINERVA"  # Minerva dataset
    OLYMPIAD_BENCH = "OLYMPIAD_BENCH"  # Olympiad benchmark problems


"""Type alias for either training or testing dataset types."""
Dataset = Union[TrainDataset, TestDataset]


def load_dataset(dataset: Dataset) -> List[Dict[str, Any]]:
    """Load a dataset from a JSON file.

    Loads and parses a JSON dataset file based on the provided dataset enum.
    The file path is constructed based on whether it's a training or testing dataset.

    Args:
        dataset: A Dataset enum value specifying which dataset to load.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset records.
            Each dictionary represents one example in the dataset.

    Raises:
        ValueError: If the dataset file cannot be found, contains invalid JSON,
            or encounters other file access errors.

    Example:
        >>> load_dataset(TrainDataset.AIME)
        [{'problem': 'Find x...', 'solution': '42', ...}, ...]
    """
    dataset_name = dataset.value.lower()
    data_dir = "train" if isinstance(dataset, TrainDataset) else "test"

    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(data_dir, f"{dataset_name}.json")
    file_path = os.path.join(current_dir, file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Error loading dataset: {exc}") from exc


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


# Define prompt templates as constants
INSTRUCTION_FOLLOWING = "Please output the final answer within \\boxed{}."

ZERO_STYLE_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here.

User: {{prompt}} Please put your answer in \\boxed{} tags.
Assistant: <think>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/bigmath_filtered_mar21",
        help="Directory to save the processed data files. Will be modified based on other parameters.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=None,
        help="Number of samples to use from the training dataset. If None, use all samples.",
    )
    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=None,
        help="Number of samples to use from the test datasets. If None, use all samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when sampling data.",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["zero_style", "instruction"],
        default="zero_style",
        help="Prompt style to use: 'zero_style' for think-then-answer format or 'instruction' for direct instruction format.",
    )

    args = parser.parse_args()

    # Get the base directory and filename
    output_dir_base = os.path.dirname(args.output_dir)
    output_dir_name = os.path.basename(args.output_dir)

    # Modify output directory name based on prompt style
    if args.prompt_style == "instruction":
        output_dir_name = f"{output_dir_name}_instruction_style"

    # Modify output directory name based on train sample size
    if (args.train_sample_size / 1000) % 1 != 0:
        size_str = f"{args.train_sample_size / 1000:.1f}k"
    else:
        size_str = f"{args.train_sample_size // 1000}k"

    if args.train_sample_size is not None:
        if "_" in output_dir_name:
            name_parts = output_dir_name.split("_", 1)
            output_dir_name = f"{name_parts[0]}{size_str}_{name_parts[1]}"
        else:
            output_dir_name = f"{output_dir_name}{size_str}"

    # Set the final output directory
    args.output_dir = os.path.join(output_dir_base, output_dir_name)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data_source = "SDSB/big_math_partial_mar21_filtered_basic"
    test_data_sources = [
        "nanoverl/minerva",
        "SDSB/aime_repeated_8x",
        "SDSB/amc_repeated_4x",
        "nanoverl/olympiad_bench",
        "nanoverl/math",
    ]

    # Get the Hugging Face datasets cache directory
    cache_dir = datasets.config.HF_DATASETS_CACHE

    # Check train dataset cache status before loading
    train_dataset_path = os.path.join(cache_dir, f"{train_data_source}")
    if os.path.exists(train_dataset_path):
        print(
            f"Train dataset '{train_data_source}' will be loaded from cache: {train_dataset_path}"
        )
    else:
        print(
            f"Train dataset '{train_data_source}' will be downloaded from Huggingface (not in cache)"
        )

    print(f"Loading the {train_data_source} dataset...", flush=True)
    train_dataset = datasets.load_dataset(
        train_data_source, trust_remote_code=True, split="train"
    )

    # Check test datasets cache status before loading
    print(f"Checking cache status for test datasets...")
    for test_data_source in test_data_sources:
        test_dataset_path = os.path.join(cache_dir, f"{test_data_source}")
        if os.path.exists(test_dataset_path):
            print(
                f"Test dataset '{test_data_source}' will be loaded from cache: {test_dataset_path}"
            )
        else:
            print(
                f"Test dataset '{test_data_source}' will be downloaded from Huggingface (not in cache)"
            )

    print(f"Loading the test datasets...", flush=True)
    test_datasets = [
        datasets.load_dataset(test_data_source, trust_remote_code=True, split="test")
        for test_data_source in test_data_sources
    ]

    # Define prompts based on style - now using constants
    if args.prompt_style == "zero_style":
        prompt = ZERO_STYLE_PROMPT_TEMPLATE
    else:  # instruction style
        prompt = None  # Not used in instruction style

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = example.pop("problem")
            answer = example.pop("answer")

            if args.prompt_style == "zero_style":
                # Zero-shot style with think-then-answer format
                data = {
                    "data_source": data_source,
                    "prompt": [],
                    "raw_prompt": prompt.replace("{{prompt}}", question),
                    "ability": "math",
                    "apply_chat_template": False,
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {"split": split, "index": idx},
                }
            elif args.prompt_style == "instruction":
                # Instruction style with direct chat format
                question_with_instruction = question + " " + INSTRUCTION_FOLLOWING
                data = {
                    "data_source": data_source,
                    "prompt": [{"role": "user", "content": question_with_instruction}],
                    "ability": "math",
                    "apply_chat_template": True,
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "split": split,
                        "index": idx,
                        "question": question_with_instruction,
                    },
                }
            else:
                raise ValueError(f"Invalid prompt style: {args.prompt_style}")

            if idx == 0:
                print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
                print(data)
            return data

        return process_fn

    train_data = train_dataset.map(
        function=make_map_fn("train", train_data_source), with_indices=True
    )

    # Sample the training data if sample size is specified
    if args.train_sample_size is not None:
        # Use seed for deterministic sampling
        train_indices = list(range(len(train_data)))
        random.shuffle(train_indices)
        train_indices = train_indices[: min(args.train_sample_size, len(train_data))]
        train_data = train_data.select(train_indices)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    train_data.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    print(f"train data size:", len(train_data))
    print(train_data[0])

    for test_data_source, test_data in zip(test_data_sources, test_datasets):
        process_fn = make_map_fn("test", test_data_source)
        test_data = test_data.map(process_fn, with_indices=True)

        # Sample the test data if sample size is specified
        if args.test_sample_size is not None:
            # Use seed for deterministic sampling
            test_indices = list(range(len(test_data)))
            random.shuffle(test_indices)
            test_indices = test_indices[: min(args.test_sample_size, len(test_data))]
            test_data = test_data.select(test_indices)

        dataset_name = os.path.basename(test_data_source.lower())
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(args.output_dir, f"{dataset_name}.parquet"))
        print(f"test data size: ({dataset_name})", len(test_df))

    print(f"Done. Data saved to {args.output_dir}")
