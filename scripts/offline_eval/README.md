# Evaluation Suite

We provide a evaluation suite across benchmarks. We tested and derived consistent results with those reported by the official DeepSeek and Qwen repositories.

## Quick Start

- Step 1 - Preprocess the datasets
    
    Download the test data from [LLM360 huggingface](https://huggingface.co/datasets/LLM360/guru_RL) and put under `SHARED_DATA_PATH`. (TODO: will update a script version directly using data from huggingface download soon).
    
- Step 2 - Generate responses and calculate metrics
    
    ```bash
    sbatch scripts/example_multinode_eval_guru7b.sh
    ```
    
    The model generations and the corresponding evaluation metrics will be saved by default saved at `./evaluation_results/test_offline_leaderboard_output/`. Please change the *data paths* in the template script upon needs.

(TODO: update a sinlge-node template script)

## Details

- Data Preprocess
    
    We also provide off-the-shelf scripts for processing a single dataset in case you want to process them individually. They are provided in the subfolders in `data_preprocess` To use them, you can simply run the `.py` file as follows. e.g.,
    
    ```python
    python data_preprocess/stem/gpqa_diamond.py
    ```

- Evaluation Script Walkthrough

    Here are some key variables you can adjust in the evaluation script.
    
    ```bash
    # leaderboard list (note, the name should be included in the test filename)
    leaderboard_list=(
      "aime"           # math
      "math"           # math
      "humaneval"      # code
      "mbpp"           # code
      "livecodebench"  # code
      "gpqa_diamond"   # science
      "supergpqa"      # science
      "arcagi"         # logic
      "zebra_puzzle"   # logic
      "codeio"         # simulation
      "cruxeval-i"     # simulation
      "cruxeval-o"     # simulation
      "finqa"          # tabular
      "hitab"          # tabular
      "multihier"      # tabular
      "livebench_reasoning"     # ood
      "livebench_language"      # ood
      "livebench_data_analysis" # ood
      "ifeval"                  # ood
    )
    
    # gpu
    n_nodes=1
    n_gpus_per_node=8
    gpu_ids=0,1,2,3,4,5,6,7
    
    # model
    model_path=<model-path>
    data_folder=<eval-data-folder>
    save_folder=<eval-output-folder>
    
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


## Customized Evaluation

### Step 1 - Prepare Your Data Preprocessing

Refer to [prepare_data](README.md#add-a-new-dataset-for-training-or-evaluation)
) for details.

### Step 2 - Define Your Reward/Score Function

Refer to [reward_function](README.md#add-a-new-dataset-for-training-or-evaluation) for details.

### Step 3 - Generate Responses & Calculate Metric

We leverage verl's rollout engine to generate responses for each test samples from all benchmarks reported in Guru.
We also use verl's reward function to calculate the metric for each benchmark.
Specifically, we implement such pipeline in `verl/trainer/main_generation.py` and `verl/trainer/main_eval.py`.
