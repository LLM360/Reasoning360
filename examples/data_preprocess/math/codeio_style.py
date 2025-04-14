import re
import os
import datasets
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
import argparse

import json
import copy

def get_dataset(local_dir, download=False):
    data_path = os.path.abspath(os.path.join(local_dir, "PythonEdu-Reasoning.jsonl"))
    if download: 
        if os.path.exists(data_path):
            pass
        else:
            url = "https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning-Raw/resolve/main/0_368500_filtered_v2_ds25.sced.jsonl"
            os.system(f'wget -O {data_path} {url}')
    dataset = []
    N_code = 0
    with open(data_path, "r") as f:
        for line in f:
            N_code += 1
            data = json.loads(line)
            common_fields = {k: v for k, v in data.items() if k != "ios"} 
            # processing each I/O and input prediction/output prediction task
            for io in data["ios"]:
                dataset.append({**common_fields, "input": json.dumps(io["input"]), "output": json.dumps(io["output"]), "given_type": "input", "predict_type": "output"})
                dataset.append({**common_fields, "input": json.dumps(io["input"]), "output": json.dumps(io["output"]),  "given_type": "output", "predict_type": "input"})
    N = len(dataset)
    N_train = int(N * 0.85)
    train_dataset = dataset[:N_train]
    test_dataset = dataset[N_train:]
    print(f"Total {N_code} code samples, {N} I/O samples.")
    return Dataset.from_list(train_dataset), Dataset.from_list(test_dataset)


# Prompt from CodeI/O
RawInputPredictionPrompt = """You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: "input": <your input>, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables’ names as specified.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}
"""

InputPredictionPrompt = """You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict a feasible input without writing any code? Please first think about the reasoning process in the mind and then provide the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here. Please put your answer in \\boxed{} tags. The final answer should be in the following format: "input": <your input>, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables’ names as specified.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}
"""

OutputPredictionPrompt = """You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict the output without writing any code? Please first think about the reasoning process in the mind and then provide the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here. Please put your answer in \\boxed{} tags. The final answer should be in the following format: "output": <your output>, where <your output> should strictly match the the output requirement as specified.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}
"""

RawOutputPredictionPrompt = """You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict the output without writing any code? Please reason and put your final answer in the following json format: "output": <your output>, where <your output> should strictly match the the output requirement as specified.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}
"""


answer_template = """"{{predict_type}}": {{sol}}"""

# Input Prediction
def make_map_fn(split):

    def process_fn(example, idx):
        given_type = example.pop("given_type")
        predict_type = example.pop("predict_type")
        if predict_type == "input":
            Prompt = InputPredictionPrompt
        else:
            Prompt = OutputPredictionPrompt
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
    parser.add_argument('--local_dir', default='')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    train_dataset, test_dataset = get_dataset(args.local_dir, download=True)

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print("data source: PyEdu, saved.")
    print(f"train data size:", len(train_dataset))
    print(f"test data size:", len(test_dataset))
    makedirs(hdfs_dir)

    copy(src=local_dir, dst=hdfs_dir)