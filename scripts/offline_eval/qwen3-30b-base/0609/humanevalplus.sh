#!/bin/bash
### ============== leadboard eval config ==============

# leaderboard list (the name should match the test file name)
leaderboard_list=(
  # "aime"           # math
  # "math"           # math
  # "olympiad_bench" # math
  "humanevalplus"      # codegen
  # "mbpp"           # codegen
  # "livecodebench"  # codegen
  # "gpqa"           # stem
)

# gpu
n_nodes=1
n_gpus_per_node=8
gpu_ids=0,1,2,3,4,5,6,7

# path
data_folder=/lustrefs/users/shibo.hao/data/feng/code/Reasoning360/data/test
save_folder=/lustrefs/users/shibo.hao/data/feng/code/Reasoning360/data/test_leaderboard_output/

# model
model_path=Qwen/Qwen3-30B-A3B-Base
model_name="qwen3-30b-base_0609"  # this will be the folder name under the save_folder

# generation hyper-parameters
n_samples=1
batch_size=128
temperature=0.6
top_k=-1 # 0 for hf rollout, -1 for vllm rollout
top_p=0.95
prompt_length=1024
response_length=31744  # 32768 - 1024, Qwen3-moe only has 32768 max len
max_num_batched_tokens=65536  # 2 x context length
tensor_model_parallel_size=2
gpu_memory_utilization=0.8
### ============== leadboard eval config ==============

# Generate timestamp for unique log files
timestamp=$(date +"%Y%m%d_%H%M%S")

# Update save_folder to include model_name
model_save_folder="${save_folder}${model_name}/"

# Check if model-specific folder exists, create if it doesn't
if [ ! -d "$model_save_folder" ]; then
  mkdir -p "$model_save_folder"
  echo "Model-specific output path created: ${model_save_folder}"
else
  echo "Model-specific output path ${model_save_folder} already exists"
fi

# Create a logs directory inside model_save_folder
logs_dir="${model_save_folder}logs/"
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
domain_mappings["humanevalplus"]="codegen"
# Initialize counters for total time
total_gen_time=0
total_eval_time=0
total_time_start=$(date +%s)

for leaderboard in "${leaderboard_list[@]}"; do
    # Get the domain for this leaderboard
    domain=${domain_mappings[$leaderboard]}
    
    # Create log files with timestamp - one for generation and one for evaluation
    gen_log_file="${logs_dir}${model_name}_${leaderboard}_gen_${timestamp}.log"
    eval_log_file="${logs_dir}${model_name}_${leaderboard}_eval_${timestamp}.log"
    
    # Find the matching file in the data folder
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
    
    # Create leaderboard-specific subfolder
    leaderboard_dir="${model_save_folder}${leaderboard}/"
    if [ ! -d "$leaderboard_dir" ]; then
      mkdir -p "$leaderboard_dir"
      echo "Leaderboard directory created: ${leaderboard_dir}" | tee -a "$gen_log_file"
    fi
    
    # Update save path to use the leaderboard-specific subfolder
    save_path="${leaderboard_dir}${file_name}"
    
    echo "Processing $leaderboard: $data_file -> $save_path" | tee -a "$gen_log_file"
    
    export CUDA_VISIBLE_DEVICES=${gpu_ids}

    # Record start time for generation
    gen_start_time=$(date +%s)
    
    # Print generation configuration to log file
    {
        echo "===== GENERATION CONFIGURATION ====="
        echo "Model: $model_name ($model_path)"
        echo "Leaderboard: $leaderboard"
        echo "Input file: $data_file"
        echo "Output file: $save_path"
        echo "Temperature: $temperature"
        echo "Top-k: $top_k"
        echo "Top-p: $top_p"
        echo "Batch size: $batch_size"
        echo "Number of samples: $n_samples"
        echo "Timestamp: $timestamp"
        echo "==================================="
    } | tee -a "$gen_log_file"
    
    # Generation step with unbuffered output
    echo "Starting generation for $leaderboard at $(date)" | tee -a "$gen_log_file"
    {
        # Force Python to use unbuffered output
        PYTHONUNBUFFERED=1 python3 -u -m verl.trainer.main_generation \
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
            rollout.max_num_batched_tokens=$max_num_batched_tokens \
            rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
            rollout.gpu_memory_utilization=$gpu_memory_utilization
    } 2>&1 | tee -a "$gen_log_file"
    
    # Record end time for generation and calculate duration
    gen_end_time=$(date +%s)
    gen_duration=$((gen_end_time - gen_start_time))
    total_gen_time=$((total_gen_time + gen_duration))
    
    # Format duration as hours:minutes:seconds
    gen_duration_formatted=$(printf '%02d:%02d:%02d' $((gen_duration/3600)) $((gen_duration%3600/60)) $((gen_duration%60)))
    
    echo "Completed generation for $leaderboard at $(date)" | tee -a "$gen_log_file"
    echo "Generation time: $gen_duration_formatted ($gen_duration seconds)" | tee -a "$gen_log_file"

    # Print evaluation configuration to log file
    {
        echo "===== EVALUATION CONFIGURATION ====="
        echo "Model: $model_name ($model_path)"
        echo "Leaderboard: $leaderboard"
        echo "Input file: $save_path"
        echo "Timestamp: $timestamp" 
        echo "==================================="
    } | tee -a "$eval_log_file"

    # Record start time for evaluation
    eval_start_time=$(date +%s)
    
    # Evaluation step with unbuffered output
    echo "Starting evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
    {
        PYTHONUNBUFFERED=1 python3 -u -m verl.trainer.main_eval \
            data.path="$save_path" \
            data.prompt_key=prompt \
            data.response_key=responses \
            data.data_source_key=data_source \
            data.reward_model_key=reward_model
    } 2>&1 | tee -a "$eval_log_file"
    
    # Record end time for evaluation and calculate duration
    eval_end_time=$(date +%s)
    eval_duration=$((eval_end_time - eval_start_time))
    total_eval_time=$((total_eval_time + eval_duration))
    
    # Format duration as hours:minutes:seconds
    eval_duration_formatted=$(printf '%02d:%02d:%02d' $((eval_duration/3600)) $((eval_duration%3600/60)) $((eval_duration%60)))
    
    echo "Completed evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
    echo "Evaluation time: $eval_duration_formatted ($eval_duration seconds)" | tee -a "$eval_log_file"
    
    # Calculate total time for this leaderboard
    task_total_time=$((gen_duration + eval_duration))
    task_total_formatted=$(printf '%02d:%02d:%02d' $((task_total_time/3600)) $((task_total_time%3600/60)) $((task_total_time%60)))
    
    echo "Total time for $leaderboard: $task_total_formatted ($task_total_time seconds)" | tee -a "$gen_log_file" | tee -a "$eval_log_file"
    
    # Add to timing summary
    echo "$leaderboard,$gen_duration,$eval_duration,$task_total_time" >> "$timing_summary"
    
    echo "Completed processing $leaderboard. Generation log: $gen_log_file, Evaluation log: $eval_log_file"
done

# Calculate overall total time
total_time_end=$(date +%s)
total_time=$((total_time_end - total_time_start))
total_time_formatted=$(printf '%02d:%02d:%02d' $((total_time/3600)) $((total_time%3600/60)) $((total_time%60)))

# Echo overall timing summary to console
echo ""
echo "===== TIMING SUMMARY ====="
echo "Total generation time: $(printf '%02d:%02d:%02d' $((total_gen_time/3600)) $((total_gen_time%3600/60)) $((total_gen_time%60))) ($total_gen_time seconds)"
echo "Total evaluation time: $(printf '%02d:%02d:%02d' $((total_eval_time/3600)) $((total_eval_time%3600/60)) $((total_eval_time%60))) ($total_eval_time seconds)" 
echo "Total run time: $total_time_formatted ($total_time seconds)"
echo "=========================="

echo "Evaluation complete."
echo "Total run time: $total_time_formatted"