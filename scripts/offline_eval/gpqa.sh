#!/bin/bash
### ============== leadboard eval config ==============

# leaderboard list (the name should math the test file name)
leaderboard_list=(
  # "aime"           # math
  # "math"           # math
  # "olympiad_bench" # math
  # "humaneval"      # codegen
  # "mbpp"           # codegen
  # "livecodebench"  # codegen
  "gpqa"           # stem
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
response_length=8192
tensor_model_parallel_size=2
gpu_memory_utilization=0.8
### ============== leadboard eval config ==============

# Extract model name from the path
model_name=$(basename "$model_path")

# Check if leaderboard generation folder exists, create if it doesn't
if [ ! -d "$save_folder" ]; then
  mkdir -p "$save_folder"
  echo "Leaderboard output path created: ${save_folder}"
else
  echo "Leaderboard output path ${save_folder} already exists"
fi

# Create a logs directory inside save_folder if it doesn't exist
logs_dir="${save_folder}logs/"
if [ ! -d "$logs_dir" ]; then
  mkdir -p "$logs_dir"
  echo "Logs directory created: ${logs_dir}"
fi

# Define domain mappings for each leaderboard
declare -A domain_mappings
domain_mappings["humaneval"]="codegen"
domain_mappings["livecodebench"]="codegen"
domain_mappings["mbpp"]="codegen"
domain_mappings["aime"]="math"
domain_mappings["math"]="math"
domain_mappings["minerva"]="math"
domain_mappings["olympiad_bench"]="math"
domain_mappings["gpqa"]="stem"

for leaderboard in "${leaderboard_list[@]}"; do
    # Get the domain for this leaderboard
    domain=${domain_mappings[$leaderboard]}
    
    # Create log files - one for generation and one for evaluation
    gen_log_file="${logs_dir}${model_name}_${leaderboard}_gen.log"
    eval_log_file="${logs_dir}${model_name}_${leaderboard}_eval.log"
    
    # Find the matching file in the data folder
    # Adjust the pattern to match your actual file naming scheme
    if [ "$leaderboard" == "olympiad_bench" ]; then
        file_pattern="${domain}__${leaderboard}_*.parquet"
    else
        file_pattern="${domain}__${leaderboard}_*.parquet"
    fi
    
    # Use find to get the actual file path
    data_file=$(find "$data_folder" -name "$file_pattern" -type f | head -n 1)
    
    if [ -z "$data_file" ]; then
        echo "No file found matching pattern: $file_pattern. Skipping." | tee -a "$gen_log_file"
        continue
    fi
    
    # Extract the file name without path
    file_name=$(basename "$data_file")
    save_path="${save_folder}${file_name}"
    
    echo "Processing $leaderboard: $data_file -> $save_path" | tee -a "$gen_log_file"
    
    export CUDA_VISIBLE_DEVICES=${gpu_ids}

    # Generation step with tee to generation log file
    echo "Starting generation for $leaderboard at $(date)" | tee -a "$gen_log_file"
    {
        python3 -m verl.trainer.main_generation \
            trainer.nnodes=$n_nodes \
            trainer.n_gpus_per_node=$n_gpus_per_node \
            data.path="$data_file" \
            data.prompt_key=prompt \
            data.n_samples=$n_samples \
            data.batch_size=$batch_size \
            data.output_path="$save_path" \
            model.path=$model_path \
            +model.trust_remote_code=True \
            rollout.temperature=$temperature \
            rollout.top_k=$top_k \
            rollout.top_p=$top_p \
            rollout.prompt_length=$prompt_length \
            rollout.response_length=$response_length \
            rollout.max_num_batched_tokens=16384 \
            rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
            rollout.gpu_memory_utilization=$gpu_memory_utilization
    } 2>&1 | tee -a "$gen_log_file"
    echo "Completed generation for $leaderboard at $(date)" | tee -a "$gen_log_file"

    # Evaluation step with tee to evaluation log file
    echo "Starting evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
    {
        python3 -m verl.trainer.main_eval \
            data.path="$save_path" \
            data.prompt_key=prompt \
            data.response_key=responses \
            data.data_source_key=data_source \
            data.reward_model_key=reward_model # this indicates key "reference" in the reward model data is the ground truth
    } 2>&1 | tee -a "$eval_log_file"
    echo "Completed evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
    
    echo "Completed processing $leaderboard. Generation log: $gen_log_file, Evaluation log: $eval_log_file"
done