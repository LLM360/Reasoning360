"""Downloads, processes, and saves PrimeIntellect datasets."""

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl.utils.data_process.prompt import build_zero_style_prompt
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter
from verl.utils.reward_score.coder1 import code_exec, remote_check_stdio, fuzzy_equal, extract_code_from_string

# Define a constant for the empty/failed example
EMPTY_EXAMPLE = {
    "data_source": None,
    "prompt": None,
    "raw_prompt": None,
    "apply_chat_template": False,
    "ability": None,
    "reward_model": None,
    "extra_info": None
}

def get_datasets(cache_dir: str):
    """
    Loads the PrimeIntellect dataset.
    """
    try:
        dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect", split="train", cache_dir=cache_dir)
        
        # TODO: Remove this line before production - only selecting 100 examples for debugging
        # dataset = dataset.select(range(100))
        
        print(f"Dataset: {len(dataset)} examples")
        return dataset, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def make_map_fn(split: str, data_source: str, prompt_style: str="zero_style") -> callable:
    def process_fn(example, idx):
        # Get the problem description
        prompt = example["problem"]
        
        # Get the solution code
        solution = example["solutions"][0] if example["solutions"] else ""
        solution = extract_code_from_string(solution)
        
        # Process tests
        tests = json.loads(example["tests"])
        if not tests:
            print(f"No tests found for example {idx}")
            return EMPTY_EXAMPLE
                
        # Handle different test types
        if tests[0]["type"] == "function_call":
            # Function call tests
            fn_name = tests[0]["fn_name"]
            test_code = f"""\
def check_{fn_name}():
"""
            for test in tests:
                input_args = ", ".join([
                    repr(arg) if isinstance(arg, (str, list, tuple, dict)) else str(arg)
                    for arg in test["input"]
                ])
                expected_output = repr(test["output"][0]) if isinstance(test["output"][0], (str, list, tuple, dict)) else test["output"][0]
                test_code += f"""    assert {fn_name}({input_args}) == {expected_output}
"""
            test_code += f"""
check_{fn_name}()
"""
            
            # Validate the solution
            full_code = f"{solution}\n{test_code}"
            succ, err = code_exec(full_code)
            if not succ:
                print(f"Test code failed for example {idx}")
                print(f"Error: {err}")
                return EMPTY_EXAMPLE
                
            oracle = json.dumps({"functional": test_code})
            
        elif tests[0]["type"] == "stdin_stdout":
            # STDIN/STDOUT tests
            stdin_list = []
            stdout_list = []
            for test in tests:
                stdin_list.append(test["input"])
                stdout_list.append(test["output"])
            
            # Validate the solution
            with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
                futures = []
                for stdin, stdout in zip(stdin_list, stdout_list):
                    futures.append(
                        executor.submit(
                            remote_check_stdio,
                            solution,
                            stdin,
                            stdout,
                        )
                    )
                for future in as_completed(futures):
                    exec_succ, output, stdin, stdout = future.result()
                    pass_test = exec_succ and fuzzy_equal(output.strip(), stdout.strip(), verbose=False)
                    if not pass_test:
                        print(f"Test code failed for example {idx}")
                        print(f"Input: {stdin}")
                        print(f"Expected output: {stdout}")
                        print(f"Actual output: {output}")
                        return EMPTY_EXAMPLE
            
            oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
        else:
            print(f"Unknown test type: {tests[0]['type']} for example {idx}")
            return EMPTY_EXAMPLE
        
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
                "ground_truth": oracle,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": solution,
                "prompt": prompt,
                "dataset": "PrimeIntellect",
                "function_name": fn_name if tests[0]["type"] == "function_call" else None,
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save PrimeIntellect datasets.")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save the processed datasets.')
    parser.add_argument('--domain', default="codegen",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="primeintellect",
                        help='Name of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=None,
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--prompt-style', type=str, choices=['zero_style'], default='zero_style',
                        help='Prompt style to use (currently only zero_style supported).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    train_output_dir = os.path.join(args.data_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    
    # Load the dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    dataset, _ = get_datasets(cache_dir)

    # Process the dataset
    process_fn = make_map_fn('train', data_source, args.prompt_style)
        
    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=64)

    # Filter out examples where processing failed
    dataset = dataset.filter(lambda x: x["data_source"] == data_source)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        dataset = dataset.filter(lambda x: length_filter.check(x),)
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Sample the dataset using utility function
    train_dataset = sample_dataset(dataset, args.train_sample_size)

    # Save the dataset using utility function
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=data_source,
        sample_size=len(train_dataset)
    )

    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Train data saved to {train_output_path} ({len(train_dataset)} samples)")