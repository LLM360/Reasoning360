# Reasoning360

This is the official repository of **Reasoning360** aiming to produce strong and provide fully-open researhc on large reasoning models, currently containing data processing and filtering, RL training, and evaluation suite. It's initialized from [verl](https://github.com/volcengine/verl).

## News
+ The ready-to-train 92K Guru RL data across six domains is released under [LLM360 huggingface](https://huggingface.co/datasets/LLM360/guru_RL)!


---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [RL Training](#rl-training)
  - [(1) Download data](#1-download-data)
  - [(2) [Optional] Customize chat template](#2-optional-customize-chat-template)
  - [(3) Train](#3-train)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
  - [Add a new dataset into RL](#add-a-new-dataset-into-rl)
  - [Pre-commit](#pre-commit)
  - [Pull Request](#pull-request)


---

## Installation

```bash
conda create -n Reasoning360 python=3.12
conda activate Reasoning360
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit cuda-nvcc
pip install uv # using uv to install packages is faster than pip
uv pip install torch==2.6.0
uv pip install flash-attn==2.7.3 --no-build-isolation
uv pip install -e .[gpu,math]
```

Alternatively, you can refer to verl [installment guidance](https://verl.readthedocs.io/en/latest/index.html) for setup.


---
## Data preparation
The full ready-to-train 92K Guru RL data is already released under [LLM360 huggingface](https://huggingface.co/datasets/LLM360/guru_RL)!  If you would like to build (or experience) the data pipeline from scratch, we also provide detailed guidances for [data preparation](./data_preprocess/README.md) and [filtering by data difficulty levels](./model_filtering/README.md).

Quick data check:
```python
import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("LLM360/guru_RL")
train_data = dataset["train"]

print(f"Columns: {train_data.column_names}")
print(json.dumps(train_data[0], indent=2))
```

---
## RL Training
### (1) Download data
Download the data and prepare them into `.parquet`, the default format in training script. (TODO: will update a script version directly using data from huggingface download soon)

### (2) [Optional] Customize chat template
Run `tools/change_tokenizer_config.py` if you want to apply '<think>'-aware chat template. Now only the 'Qwen' families are supported.
```python
python tools/change_tokenizer_config.py -i <input_model_directory> -o <output_model_directory>
```

### (3) Train
We provide the multi-node training slurm script using a `math3k` subset data for ablation, not the full data. Change the `SHARED_DATA_PATH` upon your data path.
```bash
sbatch scripts/train/example_multinode_rl_qwen32b_base.sh
```

If you need to train on the full data or include STEM data in Guru, host the llm-as-verifier model first before launching the trianing.
```bash
sbatch scripts/tools/serve_llm_as_verifier.sh
```
Then fill in the `export STEM_LLM_JUDGE_URL="<STEM_LLM_JUDGE_URL>"` by the llm-as-verifier server ip. It uses one GPU node to serve a 1.5B [general-verifier](https://huggingface.co/TIGER-Lab/general-verifier) now.


---
## Evaluation
We provide a evaluation suite of of 17 tasks supporting multi-node inference based on [verl](https://github.com/volcengine/verl). To quick start, run
```bash
sbatch scripts/offline_eval/example_multinode_eval_guru7b.sh
```
Please refer to `scripts/offline_eval/README.md` if you would like to customize more.

---
## Contributing
### Add a new dataset into RL

**Step1: Data preprocessing script**

In preprocessing, we will process the data into a list of dictionaries, and then save it into a parquet file.

1. Prompt preprocessing

    We need to process the raw question into a prompt ready to be fed to the LLM. An example is [[1](data_preprocess/math/dapo_or1_merge_deduped.py)].

    Each data point is processed into a dict, and we need to specify the prompt within the data dict:
    ```
    "prompt": [{
        "role": "user",
        "content": prompt
    }],
    ```

    Note that, when we use verl to train the model, it will turn into a prompt string with `apply_chat_template`.

    Note that:
    - You will probably need to add some task-specific instruction in the `question`. E.g., for math, we concatenate the raw problem with `Please output the final answer within \\boxed{}.`, so that it's easy to extract the answer from model output.
    - You don't need to instruct the model to "think step by step" or "wrap your thinking process in `<think>` `<\think>`". This should be taken care by verl during training with `apply_chat_template`. To enable this, we have a [script](scripts/tools/change_tokenizer_config.py) to modify the chat template of a huggingface model (currently only tested on Qwen).
    - Please add an instruction under the README of `data_preprocess`

2. Reward function

    We need to specify the information regarding reward calculation for the new dataset.

    This typically includes three keys in the dict: `data_source`, `reward_model["ground_truth"]`, `extra_info`.

    In our training, we use [`default_compute_score`](verl/utils/reward_score/__init__.py#L17), which routes the reward computing to a specific reward function implementation based on `data_source`. `ground_truth` and `extra_info` will be passed as arguments.

**Step2: Reward function**

Please look at [`default_compute_score`](verl/utils/reward_score/__init__.py#L17). You can write your own reward function for the task, and import it here. It's highly recommended to add a timeout module to avoid the training being stuck by a corner case of reward function ([example](verl/utils/reward_score/zebra_puzzle.py)).

**Step3: Training script**

Verify the inclusion of a new dataset by actually training models with it. Please refer the template script in this repo.

### Pre-commit

We use pre-commit to enforce code formatting. Before committing, make sure you have run the pre-commit checks.
```bash
pre-commit install
pre-commit run --all-files
```

### Pull Request

Please make a pull request including the data preprocessing script, reward function, and the training script.

