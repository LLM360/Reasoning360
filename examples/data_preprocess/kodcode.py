import pandas as pd
import numpy as np
import json
import os


KODCODE_DIR = "/data/samuel/datasets/kodcode"
# huggingface-cli download "KodCode/KodCode-V1" --repo-type dataset --local-dir [KODCODE_DIR]


def map_row(row):
  question = row["question"]
  solution = row["solution"]
  tests = row["test_code"]
  # tests = row["test_code"].replace("\n", "\\n").replace('"', '\\"')

  if "from solution import" in tests:
    ground_truth = {"solution_file": tests}
  else:
    ground_truth = {"functional": tests}

  # unit tests could depend on an import of solution
  # if there is no "from solution import ...", then you want to keep it fuctional instead of solution_file

  return pd.Series({
    "task_id": f"kodcode0-{row.name}", 
    "prompt": np.array([
      {'role': 'system', 'content': 'You are a helpful programming assistant. The user will ask you a question and you as the assistant solve it. The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively.'},
      {'role': 'user', 'content': f'Solve the programming task below in a Python markdown code block.\n{question}'}
    ]), 
    "entry_point": None, 
    "test": None, 
    "completion": None, 
    "examples": None, 
    "src": None, 
    "meta": None, 
    "data_source": "code", # important for using correct reward function
    "ability": "coding", 
    "reward_model": {
      "ground_truth": json.dumps(ground_truth),
      "style": "rule"
    }, 
    "extra_info": {
      "dataset": "kodcode",
      "index": row.name,
      "prompt": f'Solve the programming task below in a Python markdown code block.\n{question}',
      "reference": solution,
      "split": "train"
    }
  })


dataframes = []
for idx in range(14):
  file_name = f"train-000{str(idx).zfill(2)}-of-00014.parquet"
  print(file_name)
  
  train_kodcode = pd.read_parquet(os.path.join(KODCODE_DIR, "data", file_name))
  print(f" - df loaded {train_kodcode.shape}")

  train_kodcode_mapped = train_kodcode.apply(map_row, axis=1)
  print(f" - df mapped {train_kodcode.shape} -> {train_kodcode_mapped.shape}")

  dataframes.append(train_kodcode_mapped)


print("combining dataframes")
dataframes_combined = pd.concat(dataframes, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
train_end = int(0.8 * len(dataframes_combined))
kodcode_train = dataframes_combined[:train_end]
kodcode_test = dataframes_combined[train_end:]


print("saving dataframes")
if not os.path.exists(os.path.join(KODCODE_DIR, "processed")):
  os.mkdir(os.path.join(KODCODE_DIR, "processed"))
kodcode_train.to_parquet(os.path.join(KODCODE_DIR, "processed", "train.parquet"))
kodcode_test.to_parquet(os.path.join(KODCODE_DIR, "processed", "test.parquet"))