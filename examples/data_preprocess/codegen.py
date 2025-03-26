"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
Thanks to https://github.com/ganler/code-r1/blob/main/examples/data_preprocess/coder1.py
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from rich.rule import Rule
import rich
import matplotlib.pyplot as plt
import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.coder1 import (
    code_exec,
    remote_check_stdio,
    _ERROR_MSG_PREFIX,
)

from examples.data_preprocess.code_utils import *

WORKDING_DIR = os.path.join(os.environ.get("HOME"), "Reasoning360")


def kodcode():  # Thanks!!! to Zhangchen and Yueqin
    # library requirements?
    rich.print(Rule("Loading KodCode/KodCode-V1-SFT-R1..."))
    dataset = load_dataset("KodCode/KodCode-V1-SFT-R1")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    def make_map_fn(split):

        def process_fn(example, idx):
            reference_solution = example["solution"]
            test_code = "from solution import *\n" + example["test"].strip()
            # Filtering...
            # + block libs are used in reference solution and test code
            # + block usages are used in reference solution or test code
            # + filter out too long prompts
            # + filter out easy problems
            # + filter out failed unittests
            filter_info = {}
            for lib in BLOCK_LIBS:
                if (
                    f"import {lib}"
                    in reference_solution  # naive import detection; ast then detect would be better
                    or f"from {lib}" in reference_solution
                ):
                    print("===========Blocked lib in solution===========")
                    print(f"reference_solution:")
                    print(reference_solution)
                    print(f"lib: {lib}")
                    print(f"question_id: {example['question_id']}")
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_lib",
                            "detail": lib,
                        },
                    }
            for lib in BLOCK_LIBS:
                if f"import {lib}" in test_code or f"from {lib}" in test_code:
                    print("===========Blocked lib in test===========")
                    print(f"test_code:")
                    print(test_code)
                    print(f"lib: {lib}")
                    print(f"question_id: {example['question_id']}")
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_lib",
                            "detail": lib,
                        },
                    }
            for usage in BLOCK_USAGES:
                if usage in reference_solution:
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_usage",
                            "detail": usage,
                        },
                    }
            for usage in BLOCK_USAGES:
                if usage in test_code:
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_usage",
                            "detail": usage,
                        },
                    }
            if (
                len(tokenizer.encode(example["question"])) > MAX_PROMPT_LENGTH - 200
            ):  # -200 for (approximately) the prompt template extra tokens
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "prompt_too_long", "detail": None},
                }
            if example["gpt_difficulty"] == "easy":
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "easy_problem", "detail": None},
                }
            succ, err = code_exec(code=reference_solution, pytest=test_code)
            if not succ:
                # The above code is using the `rich` library in Python to print a formatted message in the console.
                # The message is in red color and includes the value of `example['conversation_id']`.
                # rich.print(
                #     f"[bold red]Test code failed for {example['conversation_id']}"
                # )
                print("===========Unittest failed===========")
                print(f"reference_solution:")
                print(reference_solution)
                print(f"test_code:")
                print(test_code)
                print(f"err:")
                print(err)
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "failed_unittests", "detail": None},
                }

            prompt = f"Please solve the programming task below in Python. Code should wrapped in a markdown code block.\n\n{example['question'].strip()}"
            if example["test_info"]:
                prompt += f"\n\nNote that the output function should be {str(example['test_info']).strip()}."

            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"pytest": test_code}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": reference_solution,
                    "prompt": prompt,
                    "dataset": "KodCode/KodCode-V1-SFT-R1",
                    "question_subset": example["subset"],
                    "question_id": example["question_id"],
                    "gpt_difficulty": example["gpt_difficulty"],
                },
                "filter_info": None,
            }

        return process_fn

    # shuffle the dataset, and pick 5k examples for debugging
    dataset = dataset["train"].shuffle(seed=666).select(range(150000))
    # Preprocess the dataset
    print("Executing tests to ensure correctness...")
    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
    )
    # Analyze the filter reasons
    filter_counts, filter_counts_fine_block_libs = {}, {}
    for entry in dataset:
        if entry["filter_info"] is None:
            continue
        filter_type = entry["filter_info"]["type"]
        if filter_type is not None:
            # Filter type distribution
            if filter_type not in filter_counts:
                filter_counts[filter_type] = 0
            filter_counts[filter_type] += 1
            # Filter detail distribution
            filter_detail = entry["filter_info"].get("detail", None)
            if filter_detail is not None:
                if filter_detail not in filter_counts_fine_block_libs:
                    filter_counts_fine_block_libs[filter_detail] = 0
                filter_counts_fine_block_libs[filter_detail] += 1
        # entry["filter_info"] = None

    print(f"Filtered samples from KodCode: {filter_counts}")

    plot_hist(
        filter_counts,
        file_path=os.path.join(WORKDING_DIR, "artifacts", "filter_counts.png"),
        title="Filter Sample Distribution from KodCode",
        xlabel="Filter Reason",
        ylabel="Count",
    )
    plot_hist(
        filter_counts_fine_block_libs,
        file_path=os.path.join(
            WORKDING_DIR, "artifacts", "filter_counts_fine_block_libs.png"
        ),
        title="Blocked Library Distribution from KodCode",
        xlabel="Blocked Library",
        ylabel="Count",
    )

    dataset = dataset.filter(lambda x: x["data_source"] is not None)
    print(f"Remaining samples from KodCode: {len(dataset)}")

    # pick random 50k examples for RL, otherwise it's too large
    dataset = dataset.select(range(50000 + N_TESTSET_PER_DATASET))

    # Split into train and test
    # splits = dataset["train"].train_test_split(
    #     test_size=N_TESTSET_PER_DATASET, seed=666
    # )
    splits = dataset.train_test_split(test_size=N_TESTSET_PER_DATASET, seed=666)
    train_dataset = splits["train"].shuffle(seed=666)
    test_dataset = splits["test"]
    return train_dataset, test_dataset


# this dataset is super noisy and needs code execution to verify the tasks
def taco():
    rich.print(Rule("Loading likaixin/TACO-verified..."))
    dataset = load_dataset("likaixin/TACO-verified")["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            oracle = json.loads(example["input_output"])
            source = example["source"]

            # skip poorly formatted examples
            if source in ["geeksforgeeks", "leetcode"]:
                return EMPTY_RETURN

            # too short description
            if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
                return EMPTY_RETURN

            # no image
            if "image" in example["question"].lower() or "\n![" in example["question"]:
                return EMPTY_RETURN

            prompt_pieces = [
                "Solve the programming task below in a Python markdown code block.",
                example["question"].strip(),
            ]
            if example["starter_code"].strip():
                prompt_pieces.append(
                    "Also feel free to reuse/extend the following starter code:"
                )
                prompt_pieces.append(
                    f"```python\n{example['starter_code'].strip()}\n```"
                )

            ##
            ## Customization
            ##
            if "fn_name" in oracle:  # the dataset is too noisy
                fn_name = oracle["fn_name"]
                if source == "leetcode":
                    fn_name = "Solution()." + fn_name

                test_code = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""

                if source in ["leetcode", "hackerrank"]:
                    test_code += f"    assert _deep_eq({fn_name}(*i), o)"
                elif source == "codewars":
                    test_code += f"    assert _deep_eq({fn_name}(*i), o[0])"
                else:
                    raise ValueError(f"Unknown source: {source}")

                _check_test = example["solutions"][-1] + "\n" + test_code
                if source in ["leetcode"]:
                    _check_test = PY_IMPORTS + _check_test

                succ, err = code_exec(_check_test)
                if not succ:
                    rich.print(f"[bold red]Test code failed for {source}")
                    print(_check_test)
                    print(err)
                    return EMPTY_RETURN
                oracle = json.dumps({"functional": test_code})
                assert example["starter_code"].strip() != ""
            elif "inputs" in oracle and "outputs" in oracle:
                stdin_list, stdout_list = minimize_stdio(
                    oracle["inputs"], oracle["outputs"]
                )
                if len(stdin_list) == 0:
                    return EMPTY_RETURN

                with ThreadPoolExecutor(
                    max_workers=min(len(stdin_list), 8)
                ) as executor:
                    futures = []
                    for stdin, stdout in zip(stdin_list, stdout_list):
                        futures.append(
                            executor.submit(
                                remote_check_stdio,
                                example["solutions"][-1],
                                stdin,
                                stdout,
                            )
                        )
                    for future in as_completed(futures):
                        exec_succ, output, stdin, stdout = future.result()
                        pass_test = exec_succ and output.strip() == stdout.strip()
                        if not pass_test:
                            rich.print(f"[bold red]Test code failed for {source}")
                            print(example["solutions"][-1])
                            print(f"{exec_succ = }")
                            print(f"{stdin = }", f"{stdout = }")
                            if output.startswith(_ERROR_MSG_PREFIX):
                                print("output = \n", output)
                            else:
                                print(f"{output = }")
                            return EMPTY_RETURN

                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            prompt = "\n".join(prompt_pieces)
            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (
                        example["solutions"][0] if example["solutions"] else ""
                    ),
                    "dataset": "likaixin/TACO-verified",
                },
            }

        return process_fn

    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        remove_columns=dataset.column_names,
    ).filter(lambda x: x != EMPTY_RETURN)
    splits = dataset.train_test_split(
        test_size=max(1, min(N_TESTSET_PER_DATASET, len(dataset) * 0.1)), seed=666
    )
    train_dataset = splits["train"]
    test_dataset = splits["test"]

    for t in dataset:
        print(f"{t = }")
        t["extra_info"]["split"] = "test"

    print(f"Taco train set: {train_dataset}")
    print(f"Taco test set: {test_dataset}")

    return train_dataset, test_dataset
    train_dataset = dataset["train"]
    test_dataset = dataset["valid"][:N_TESTSET_PER_DATASET]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            if "<image>" in example["description"]:
                print("Description includes image, skipping...")
                return EMPTY_RETURN

            stdin_list = (
                example["public_tests"]["input"]
                + example["private_tests"]["input"]
                + example["generated_tests"]["input"]
            )
            stdout_list = (
                example["public_tests"]["output"]
                + example["private_tests"]["output"]
                + example["generated_tests"]["output"]
            )

            stdin_list, stdout_list = minimize_stdio(
                stdin_list, stdout_list, max_n_tests
            )
            assert len(stdin_list) == len(stdout_list)
            if len(stdin_list) == 0:
                return EMPTY_RETURN

            prompt = (
                "Solve the programming task below in a Python markdown code block. "
                "Each time, given inputs through STDIN (like those in the 'Input' section), the program "
                "produces outputs through STDOUT (like those in the 'Output' section)."
                f"\n\n{example['description'].strip()}"
            )
            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {"inputs": stdin_list, "outputs": stdout_list}
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (
                        example["solutions"]["solution"][0]
                        if example["solutions"]["solution"]
                        else ""
                    ),
                    "dataset": "deepmind/code_contests",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=dataset.column_names,
    ).filter(lambda x: x != EMPTY_RETURN)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    test_dataset = load_dataset(
        "newfacade/LeetCodeDataset",
        data_files="LeetCodeDataset-v2-test-problems.jsonl",
    )["train"]
    print("Test set:", test_dataset)

    train_dataset = concatenate_datasets(
        [
            load_dataset(
                "newfacade/LeetCodeDataset",
                data_files="LeetCodeDataset-v1-train-problems.jsonl",
            )["train"],
            load_dataset(
                "newfacade/LeetCodeDataset",
                data_files="LeetCodeDataset-v2-train-problems.jsonl",
            )["train"],
        ]
    ).filter(
        lambda example: example["meta"]["question_id"]
        not in set([d["question_id"] for d in test_dataset["meta"]])
    )
    print("Before deduplication - Training set:", train_dataset)

    first_time_idx = []
    seen_question_ids = set()
    for i, example in enumerate(train_dataset):
        if example["meta"]["question_id"] not in seen_question_ids:
            first_time_idx.append(i)
            seen_question_ids.add(example["meta"]["question_id"])
    train_dataset = train_dataset.select(first_time_idx)

    print("After deduplication - Training set:", train_dataset)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = f"Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\n{example['meta']['query'].strip()}"
            return {
                "data_source": "code",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {
                            "functional": f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
                        }
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["completion"],  # C++?
                    "prompt": prompt,
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    print(f"Leetcode2k train set: {train_dataset}")
    print(f"Leetcode2k test set: {test_dataset}")
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=os.path.join(WORKDING_DIR, "data"))
    parser.add_argument(
        "--dataset_names", default="kodcode", help="comma separated dataset names"
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_names = args.dataset_names.split(",")

    train_datasets = []
    test_datasets = []

    dataset_map = {
        "kodcode": kodcode,
        "taco": taco,
        "leetcode2k": leetcode2k,
    }
    dataset_makes = [dataset_map[name] for name in dataset_names]
    names = "-".join([make.__name__ for make in dataset_makes])

    for train, test in [make() for make in dataset_makes]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    local_dir = os.path.join(
        root_dir, f"codegen-{round(len(train_dataset) / 1000)}k-{names}"
    )
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
