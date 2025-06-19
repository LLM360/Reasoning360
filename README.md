# Reasoning360

<p align="center">
  <a href="https://arxiv.org/abs/2506.14965">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2506.14965-b31b1b?style=flat&logo=arxiv">
  </a>
  <a href="https://huggingface.co/datasets/LLM360/guru-RL-92k">
    <img alt="Dataset" src="https://img.shields.io/badge/Data-guru--92k-blue?logo=huggingface&logoColor=yellow">
  </a>
  <a href="https://huggingface.co/LLM360/guru-7B">
    <img alt="Model" src="https://img.shields.io/badge/Model-guru--model-ffcc00?logo=huggingface&logoColor=yellow">
  </a>
</p>


This is the official repository of **Reasoning360** aiming to produce strong and provide fully-open researhc on large reasoning models, currently containing data processing and filtering, RL training, and evaluation suite. It's initialized from [verl](https://github.com/volcengine/verl).

## 🔥News
+ Our paper to analyze and improve multi-domain RL for LLM reasoning with Guru data "[Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective](https://arxiv.org/abs/2506.14965)" is out on arxiv.

+ The ready-to-train 92K Guru RL data across six domains is released under [LLM360 huggingface](https://huggingface.co/datasets/LLM360/guru_RL).


---
## Table of Contents
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [RL Training](#rl-training)
  - [(1) Download data](#1-download-data)
  - [(2) \[Optional\] Customize chat template](#2-optional-customize-chat-template)
  - [(3) \[Optional\] SandboxFusion Code Execution](#3-optional-sandboxfusion-code-execution)
  - [(4) Train](#4-train)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
  - [Add a new dataset for training (or evaluation)](#add-a-new-dataset-for-training-or-evaluation)
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
uv pip install -e .[gpu,math,vllm,test]
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
dataset = load_dataset("LLM360/guru-RL-92k")
train_data = dataset["train"]

print(f"Columns: {train_data.column_names}")
print(json.dumps(train_data[0], indent=2))
```

---
## RL Training
### (1) Download data
Download the data and prepare them into `.parquet`, the expected default format in training script. We provide a simple script to download and organize Guru data `scripts/tools/download_guru.py`, with all dataset files for training, online & offline evaluation to local directories.
By defauly, training files will be put in `./data/train`. Online evaluation files will be put in `./data/online_eval`. Offline evaluation files will be put in `./data/offline_eval`.

### (2) [Optional] Customize chat template
Run `tools/change_tokenizer_config.py` if you want to apply 'think'-aware chat template. Now only the 'Qwen' families are supported.
```python
python tools/change_tokenizer_config.py -i <input_model_directory> -o <output_model_directory>
```

### (3) [Optional] SandboxFusion Code Execution

SandboxFusion provides secure code execution for training and evaluation. It supports both containerized SLURM deployment and local installation.

#### Quick Setup

**Option 1: SLURM Container (Recommended for production)**
```bash
# Download container
enroot import docker://varad0309/code_sandbox:server

# Deploy with SLURM
sbatch scripts/sandbox/run_server.sbatch
```

**Option 2: Local Installation (Development only)**
```bash
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion
poetry install
make run-online
```

#### Configuration

Configure sandbox servers in your training script:

```bash
# Single server
export SANDBOX_FUSION_SERVERS="fs-mbz-gpu-044"

# Multiple servers (load balancing)
export SANDBOX_FUSION_SERVERS="fs-mbz-gpu-044,fs-mbz-gpu-045"
```

Or programmatically:
```python
from verl.utils.reward_score.coder1.sandboxfusion_exec import code_exec_sandboxfusion

# Single server
success, output = code_exec_sandboxfusion(
    code="print('Hello')", 
    sandbox_servers="fs-mbz-gpu-044"
)

# Multiple servers
success, output = code_exec_sandboxfusion(
    code="print('Hello')", 
    sandbox_servers=["fs-mbz-gpu-044", "fs-mbz-gpu-045"]
)
```

For detailed setup instructions, see [`verl/utils/reward_score/coder1/README.md`](verl/utils/reward_score/coder1/README.md).


### (4) Train
We provide the multi-node training slurm script using a `math3k` subset data for ablation, not the full data. Change the `SHARED_DATA_PATH` upon your data path.
```bash
sbatch scripts/train/example_multinode_rl_qwen32b_base.sh
```

If you need to train on the full data or include STEM data in Guru, host the llm-as-verifier model first before launching the training.
```bash
sbatch scripts/tools/serve_llm_as_verifier.sh
```
Then fill in the `export STEM_LLM_JUDGE_URL="<STEM_LLM_JUDGE_URL>"` by the llm-as-verifier server IP. It uses one GPU node to serve a 1.5B [general-verifier](https://huggingface.co/TIGER-Lab/general-verifier) now.

(TODO: build a single-node script not using slurm)


---
## Evaluation
We provide a evaluation suite of of 17 tasks supporting multi-node inference based on [verl](https://github.com/volcengine/verl). For quick start, run
```bash
sbatch scripts/offline_eval/example_multinode_eval_guru7b.sh
```
Please refer to `scripts/offline_eval/README.md` if you would like to know and customize evaluation details.

---
## Contributing
### Add a new dataset for training (or evaluation)

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

