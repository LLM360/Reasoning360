import re
import os
from datasets import Dataset
import argparse
import json
import random
import numpy as np
import torch
import time
import transformers

from verl.utils.data_process.filter import LengthFilter

def get_dataset(cache_dir, download=False):
    data_path = os.path.join(cache_dir, "PythonEdu-Reasoning.jsonl")
    # data_path = os.path.abspath(os.path.join(cache_dir, "PythonEdu-Reasoning.jsonl"))
    if download: 
        if os.path.exists(data_path):
            pass
        else:
            url = "https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning-Raw/resolve/main/0_368500_filtered_v2_ds25.sced.jsonl"
            os.system(f'wget -O {data_path} {url}')
    dataset = []
    N_code = 0
    if not os.path.exists(data_path):
        time.sleep(5)
    with open(data_path, "r") as f:
        for line in f:
            N_code += 1
            data = json.loads(line)
            common_fields = {k: v for k, v in data.items() if k != "ios"} 
            # processing each I/O and input prediction/output prediction task
            # print("=" * 10)
            # print(json.dumps(common_fields, indent=4))
            # print(json.dumps(data["ios"], indent=4))
            # for io in data["ios"]:
            if data["ios"]:
                io = data["ios"][0]
                dataset.append({**common_fields, "input": json.dumps(io["input"]), "output": json.dumps(io["output"]), "given_type": "input", "predict_type": "output"})
                dataset.append({**common_fields, "input": json.dumps(io["input"]), "output": json.dumps(io["output"]),  "given_type": "output", "predict_type": "input"})
    N = len(dataset)
    N_train = int(N * 0.85)
    train_dataset = dataset[:N_train]
    test_dataset = dataset[N_train:]
    print(f"Total {N_code} code samples, {N} I/O samples.")
    return Dataset.from_list(train_dataset), Dataset.from_list(test_dataset)

# Original CodeIO prompts
RawInputPredictionPrompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here.

User: You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: "input": <your input>, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables' names as specified. Please put your answer in \\boxed{} tags.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}
Assistant: <think>
"""

RawOutputPredictionPrompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here.

User: You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict the output without writing any code? Please reason and put your final answer in the following json format: "output": <your output>, where <your output> should strictly match the the output requirement as specified. Please put your answer in \\boxed{} tags.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}
Please put your answer in \\boxed{} tags.
Assistant: <think>
"""

answer_template = """"{{predict_type}}": {{sol}}"""

# Input Prediction
def make_map_fn(split):

    def process_fn(example, idx):
        given_type = example.pop("given_type")
        predict_type = example.pop("predict_type")
        if predict_type == "input":
            Prompt = RawInputPredictionPrompt
        else:
            Prompt = RawOutputPredictionPrompt
        raw_prompt = Prompt.replace("{{given_type}}", given_type)
        for key in ["problem_description", "io_requirements", given_type, "refcode"]:
            feature = example.pop(key)
            if key in ["input", "output"]:
                raw_prompt = raw_prompt.replace("{{given}}", str(feature))
            else:
                raw_prompt = raw_prompt.replace(f"{{{{{key}}}}}", str(feature))
    
        sol = example.pop(predict_type)
        answer = answer_template.replace("{{predict_type}}", predict_type)
        answer = answer.replace("{{sol}}", sol)
        data = {
            "data_source": "codeio-pyedu",
            "prompt": [],
            "raw_prompt": raw_prompt,
            "ability": "coding-inference",
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/codeio_style', help='Directory to save the processed data files')
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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get CodeIO dataset
    train_dataset, test_dataset = get_dataset(f"{os.path.expanduser('~')}/.cache/huggingface/datasets", download=True)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    # Calculate max lengths for train and test datasets
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    length_filter = LengthFilter(tokenizer=tokenizer, max_length=2048)
    train_dataset = train_dataset.filter(lambda x: length_filter.check(x))
    test_dataset = test_dataset.filter(lambda x: length_filter.check(x))
    print(f"Max train prompt length: {max([len(tokenizer.tokenize(example['raw_prompt'])) for example in train_dataset])}")
    print(f"Max test prompt length: {max([len(tokenizer.tokenize(example['raw_prompt'])) for example in test_dataset])}")
    
    if args.train_sample_size is not None:
        train_dataset = train_dataset.select(range(args.train_sample_size))
    if args.test_sample_size is not None:
        test_dataset = test_dataset.select(range(args.test_sample_size))

    # Save datasets
    train_dataset.to_parquet(os.path.join(args.output_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output_dir, 'test.parquet'))
    
    print("data source: PyEdu, saved.")
    print(f"train data size:", len(train_dataset))
    print(f"test data size:", len(test_dataset))
