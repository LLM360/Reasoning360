# Evaluation Suite

We provide our own evaluation suite for experiments, which provides consistent results with those reported by the official DeepSeek and Qwen repositories.

# Quick Start

- Step 1 - Preprocess the datasets
    
    ```python
    bash ./data_preprocess/run_process_leaderboard.sh
    ```
    
    The processed datasets will be saved at `./data` 
    
- Step 2 - Generate responses and calculate metrics
    
    ```bash
    sbatch ./scripts/offline_eval/guru/run_all.sh
    
    ```
    
    The model generations and the corresponding evaluation metrics will be saved at `./data/test_leaderboard_output`
    

# Details

- Data Preprocess
    
    We also provide off-the-shelf scripts for processing a single dataset in case you want to process them individually. They are provided in the subfolders in `./data_preprocess` To use them, you can simply run the `.py` file as follows.
    
    ```python
    python ./data_preprocess/stem/gpqa_diamond.py
    ```
    
- Response Generation & Metric Calculation
    
    We provide a template as `./scripts/offline_eval/eval_template.sh` , where you need to specify the model, benchmark, and the corresponding generation hyper-parameter in the beginning configuration section shown below.
    
    ```bash
    # leaderboard list (the name should math the test file name)
    leaderboard_list=(
      "aime"           # math
      "aime2025"       # math
      "math"           # math
      "amc"            # math
      "humaneval"      # code
      "mbpp"           # code
      "livecodebench"  # code
      "gpqa"           # science
      "supergpqa"      # science
      "arcagi"         # logic
      "zebra_puzzle"   # logic
      "codeio"         # simulation
      "cruxeval-i"     # simulation
      "cruxeval-o"     # simulation
      "finqa"          # tabular
      "hitab"          # tabular
      "multihier"      # tabular
    )
    
    # gpu
    n_nodes=1
    n_gpus_per_node=8
    gpu_ids=0,1,2,3,4,5,6,7
    
    # model
    model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    data_folder=./data/test/
    save_folder=./data/test_leaderboard_output/
    
    # generation hyper-parameters
    n_samples=1
    batch_size=128
    temperature=1.0
    top_k=-1 # 0 for hf rollout, -1 for vllm rollout
    top_p=0.7
    prompt_length=1024
    response_length=32768
    tensor_model_parallel_size=2
    gpu_memory_utilization=0.8
    ### ============== leadboard eval config ==============
    ```
    

# Customized Evaluation

### Step 1 - Prepare Your Data Preprocessing

Refer to [prepare_data](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) for details.

### Step 2 - Define Your Reward/Score Function

Refer to [reward_function](https://verl.readthedocs.io/en/latest/preparation/reward_function.html) for details.

### Step 3 - Generate Responses & Calculate Metric

We leverage verl's rollout engine to generate responses for each test samples from all benchmarks reported in Guru.
We also use verl's reward function to calculate the metric for each benchmark.
Specifically, we implement such pipeline in `./verl/trainer/main_generation.py` and `./verl/trainer/main_eval.py`.

To launch the rollout jobs, you can simply run the following command.
```bash
bash scripts/offline_eval/guru/run_all.sh
```

For other models, you can follow the same pipeline to launch the rollout jobs and evaluate the generated responses.