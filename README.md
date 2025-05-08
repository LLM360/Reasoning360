# Reasoning360

The repo is an attempt to replicate the large-scale RL training of [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1).

It's initialized from [verl](https://github.com/volcengine/verl). verl's README is appended below.

Note: When you change the core code inside verl, please add a note `# added by reasoning360` near the code change.

## Setup

```bash
conda create -n Reasoning360 python=3.12
conda activate Reasoning360
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit cuda-nvcc
pip install uv # using uv to install packages is faster than pip
uv pip install torch
uv pip install flash-attn --no-build-isolation
uv pip install -e .[gpu,math]
```

Remember to process data and wandb and huggingface login before launching the experiments.

---
## Experiment Run
The ready-to-train data, called `guru15k` now (sampled 15k for data ablation), are now available on both M1 and M2 cluster.

Make sure to run `tools/change_tokenizer_config.py` first to apply our customized chat template. Currently only the `Qwen` family is supported. See detailed usage in the argparse.

### On M1 cluster
```
sbatch scripts/train/exp_1111116/m1_<xxdata>.sh
```

### On M2 cluster
```
sbatch scripts/train/exp_1111116/m2_<xxdata>.sh
```

The `exp_1111116` means training six single domain data separately and one mixture of 6 domain data, i.e., 7 runs in total. Specifcially, the `guru15k` consists of `math`, `codegen`, `logic`, `simulation`, `table`, `stem` domains, each with 2.5k data. We use the uniform distribution for initial experiments. New ablations would be under new directories.

Note, for `codegen` and `mix` training, as M2 cluster has troubles in multiprocessing python program execution, try to train them on the M1 cluster. 

Note, for `stem` and `mix` training, as both requires llm-as-judge, run `sbatch data_preprocess/run_verifier_judge.sh` first and `export STEM_LLM_JUDGE_URL="http://{node_ip}:8000` before training. It uses one GPU node to serve a 1.5B verifier now.


---
## Difficulty filtering
See `model_filtering` for details

---
## How to add the dataset

### Data preprocessing script

In preprocessing, we will process the data into a list of dictionaries, and then save it into a parquet file.

1. Prompt preprocessing

    We need to process the raw question into a prompt ready to be fed to the LLM. An example is [[1](data_preprocess/math/bigmath_preview_filtered_mar21.py)].

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

    In our training, we use [`_default_compute_score`](verl/utils/reward_score/__init__.py#L17), which routes the reward computing to a specific reward function implementation based on `data_source`. `ground_truth` and `extra_info` will be passed as arguments.

### Reward function

Please look at [`_default_compute_score`](verl/utils/reward_score/__init__.py#L17). You can write your own reward function for the task, and import it here. It's highly recommended to add a timeout module to avoid the training being stuck by a corner case of reward function ([example](verl/utils/reward_score/zebra_puzzle.py)).

### Training script

Verify the inclusion of a new dataset by actually training models with it. It's recommended to start with Qwen2.5-3B/7B-Instruct. Please refer the template script in this repo.

Notes:

- It's recommended to download the model from huggingface before training. Please use `huggingface-cli download <repo_id> --local-dir <your_local_directory> `, and change `BASE_MODEL` to the local path in the script.

### Pull Request

Finally, please make a pull request including the data preprocessing script, revised `_default_compute_score`, and the training script.


---
## Training (docker + fsdp/megatron)
Multi-node training 7B w/ FSDP backend:
```bash
sbatch scripts/templates/docker-fsdp-qwen7b-4nodes-grpo.sh
```

Multi-node training 32B w/ FSDP backend:

(Note: as vllm updates to >= 0.8, the docker image needs update; can turn to local conda env for now)
```bash
sbatch scripts/templates/docker-fsdp-qwen32b-16nodes-grpo.sh
```

Multi-node training 32B w/ Megatron backend:
```bash
sbatch scripts/templates/docker-megatron-dsr1_qwen_32b-16nodes-ppo.sh
```

## Usage (conda+fsdp)
First export the conda binary path (can check with `which python` and `which ray`):
```
export CONDA_BIN_PATH=/path/to/conda/bin/
```

Single node, 8-H100 GPUs:
```bash
bash scripts/templates/conda-fsdp-qwen2.5_math_3b-1node-ppo.sh
```

Multi-node training 7B:
```bash
sbatch scripts/templates/conda-fsdp-qwen7b-4nodes-grpo.sh
```

Multi-node training 32B:
```bash
sbatch scripts/templates/conda-fsdp-qwen32b-16nodes-grpo.sh
```

The single-node script directly prints log in the terminal; the multi-node script prints log in the slurm log (`.out` and `.err`) files. Check wandb for the experiment logs.

Adjust the template script to fit your needs.

---
## Data viewer

1. Installation

First install streamlit with
```bash
pip install streamlit
```

2. Running

Then run 
```bash
streamlit run data_viewer.py
```

For custom data sources, you can specify arguments:
```bash
streamlit run data_viewer.py -- --source-type wandb_table --file-path path/to/your/file.json
```

Currently it only supports loading the wandb logging table (enabled with `trainer.val_generations_to_log_to_wandb=...` in a training script).

We are extending it to other data sources.

3. Viewing the data

Need to set up port forwarding
```bash
ssh -L 8501:localhost:8501 username@cluster_node
```

Then access `127.0.0.1:8501`


## Data analysis

Split a dataset, and generate multiple responses with SGLang.

```python transform_dataset.py --dataset SynthLabsAI/Big-Math-RL-Verified --output big-math-rl-verified_problems --splits 64```

install SGLang from github (need to be after this [PR](https://github.com/sgl-project/sglang/pull/3754) to avoid a bug in batch inference)

```bash data_analysis/sgl_generate.sh```

The script will automatically launch 64 jobs (each on one node) and monitor them. If some job fails unexpectedly, you can launch the job in `temp_job_scripts` and add the new job id to `sgl_job_ids.txt` so that it'll be monitored.


## Contributing

We use `yapf` to enforce strict code formatting. Before committing, make sure you have run the pre-commit checks.
```bash
pre-commit install
pre-commit run --all-files
```

---

## Trouble shooting
1. Gloo socket issue:
```bash
export GLOO_SOCKET_IFNAME=ens10f0np0
```
already added in the template scripts.

2. Training got stuck after several steps rollout phase.
We found it happens in `vllm` all-reduce operations. Setting the following
```
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
``` 
in the script would solve it.
