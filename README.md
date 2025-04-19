# Reasoning360

The repo is an attempt to replicate the large-scale RL training of [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1).

It's initialized from [verl](https://github.com/volcengine/verl). verl's README is appended below.

## Setup


```bash
conda create -n Reasoning360 python=3.12
conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit cuda-nvcc
pip install uv # using uv to install packages is faster than pip
uv pip install torch
uv pip install flash-attn --no-build-isolation
uv pip install -e .[gpu,math]
```

Remember to process data and wandb and huggingface login before launching the experiments.

---
## How to add the dataset

### Data preprocessing script

In preprocessing, we will process the data into a list of dictionaries, and then save it into a parquet file.

1. Prompt preprocessing

    We need to process the raw question into a prompt ready to be fed to the LLM.

    Depending on the specific LLM to be used, there are two kind of prompts

    - Chat style (for instruct-tuned model)

        Example: [[1]](examples/data_preprocess/math/bigmath_preview_filtered_mar21_r1_instruction_style.py)

        This is used to train a model that has been trained on long-CoT, including `LLM360/Reason-SFT-32B-v0.1`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

        In the dict, we need to specify the prompt within the data dict:
        ```             
        "prompt": [{
            "role": "user",
            "content": question
        }],
        ```

        When we use verl to train the model, it will turn into a prompt string with `apply_chat_template`.

        Note that:
        - You will probably need to add some task-specific instruction in the `question`. E.g., for math, we concatenate the raw problem with `Let's think step by step and output the final answer within \\boxed{}.`, so that it's easy to extract the answer from model output.
        - These models are not trained with system prompt, so there is no need to add system prompt.
        - The `apply_chat_template` method of these two models will automatically append `<think>` to the end of the prompt


    - R1-zero style (for base model)

        Example: [[1]](examples/data_preprocess/math/bigmath_preview_filtered_mar21_r1_zero_style.py)
    
        This is used for training from a base model (without instruction-tuned), e.g., `Qwen/Qwen2.5-32B`.

        In this case, since the model is not trained to use `<think>`, and it'll not automatically add `<think>` in `apply_chat_template`, we will need to do this processing here ourselves.

        First define a template:
        ```
        template = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here.
        User: {{prompt}} Please put your answer in \\boxed{} tags.
        Assistant: <think>"""
        ```
        Then, in the data dict, we need to specify:
        ```
        "raw_prompt": template.replace("{{prompt}}", question),
        "apply_chat_template": False,
        ```
        With these two keys, verl will not apply chat template to the `prompt`, but will directly use `raw_prompt`.

2. Reward function

    We need to specify the information regarding reward calculation for the new dataset.

    This typically includes three keys in the dict: `data_source`, `reward_model["ground_truth"]`, `extra_info`.

    In our training, we use [`_default_compute_score`](verl/utils/reward_score/__init__.py#L17), which routes the reward computing to a specific reward function implementation based on `data_source`. `ground_truth` and `extra_info` will be passed as arguments.

### Reward function

Please look at [`_default_compute_score`](verl/utils/reward_score/__init__.py#L17). You can write your own reward function for the task, and import it here.

### Training script

Verify the inclusion of a new dataset by actually training models with it.

Please refer to these two templates: [base model](scripts/templates/bigmath-qwen7b-math-4nodes-dapo.sh), [instruct model](scripts/templates/bigmath-r1-1.5b-4nodes-dapo.sh).

Notes:

- It's recommended to download the model from huggingface before training. Please use `huggingface-cli download <repo_id> --local-dir <your_local_directory> `, and change `BASE_MODEL` to the local path in the script.

### Pull Request

Finally, please make a pull request including the data preprocessing script, revised `_default_compute_score`, and the training script.



## Using existing datasets


1. Math
    Take `Deepscaler` as an example.
    Deepscaler has 40K high-quality math (Q, A) pairs from previous AIME, AMC, etc. Prepare this by running:
    ```bash
    python examples/data_preprocess/math/deepscaler_preview.py --local_dir data/deepscaler_preview
    ```

2. Code
    Take `Kodcode` as an example.
    Kodcode is a dataset of 400K code (Q, solution, utests) triplets from previous ACM ICPC, etc. Prepare this by running:
    ```bash
    python examples/data_preprocess/code/codegen.py --dataset_names kodcode
    ```

3. Logic
    Take `Ordering Puzzle` as an example.
    ```bash
    python examples/data_processs/logic/puzzle_gen.py --num_puzzles 10000 --output_dir data/puzzles_dataset --output_file puzzles_dataset.json --test True
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
