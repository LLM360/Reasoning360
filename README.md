# Reasoning360

The repo is an attempt to replicate the large-scale RL training of [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1).

It's initialized from [verl](https://github.com/volcengine/verl). verl's README is appended below.

## Setup
Check the guide of verl to setup the environment.

Remember to process data and wandb login before launching the experiments.

---
## Data preparation

### 1. Math
Take `Deepscaler` as an example.
Deepscaler has 40K high-quality math (Q, A) pairs from previous AIME, AMC, etc. Prepare this by running:
```bash
python examples/data_preprocess/deepscaler_preview.py --local_dir data/deepscaler_preview
```

### 2. Code
Take `Kodcode` as an example.
Kodcode is a dataset of 400K code (Q, solution, utests) triplets from previous ACM ICPC, etc. Prepare this by running:
```bash
python examples/data_preprocess/codgen.py --dataset_names kodcode
```

## 3. Logic
Take `Ordering Puzzle` as an example.
```bash
python examples/data_processs/puzzle_gen.py --num_puzzles 10000 --output_dir data/puzzles_dataset --output_file puzzles_dataset.json --test True
```


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

## Change reward metric
Set `reward_model.reward_metric` in the config file or cli arguments. Can choose from "prime_math", "math_verify", "boxed_math_verify" for math. Default is None, i.e., use "prime_math" for math.


### To add new dataset...


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
