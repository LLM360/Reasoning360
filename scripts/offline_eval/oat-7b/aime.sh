#!/bin/bash
#SBATCH --job-name=zhoujun-rl-guru15k-table2.5k
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/%j_%x.log
#SBATCH --error=slurm/%j_%x.log
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --qos=iq
#SBATCH --exclude=fs-mbz-gpu-[088,317,440,497]


# =================== Environment ===================
# may vary from cluster to cluster, please check the environment variables
# export LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:$LD_LIBRARY_PATH \
#        UCX_TLS=dc \
#        UCX_NET_DEVICES=mlx5_ib0:1 \
#        CUDA_DEVICE_ORDER=PCI_BUS_ID \
#        NCCL_SOCKET_IFNAME=eth0 \
#        NCCL_DEBUG=WARN \
#        NCCL_NET_GDR_LEVEL=5 \
#        NCCL_MIN_NCHANNELS=32 \
#        NCCL_TOPO_FILE=/mnt/users/runner/scripts/ndv5-topo.xml \
#        OMPI_MCA_coll_hcoll_enable=0 \
#        OMPI_MCA_plm_rsh_no_tree_spawn=1 \
#        OMPI_MCA_plm_rsh_num_concurrent=800 \
#        NCCL_IB_QPS_PER_CONNECTION=4 \
#        NCCL_P2P_NET_CHUNKSIZE=$((512*1024)) \
#        NCCL_PXN_DISABLE=1

# export UCX_NET_DEVICES=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
# export CUDA_DEVICE_MAX_CONNECTIONS=1

export NCCL_DEBUG=info
export NCCL_ALGO=NVLSTree
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"

# We'll track PIDs so we can wait on them and detect errors
declare -A pids

# Spawn each check in the background
for host in "${nodes[@]}"; do
    echo "Spawning GPU check on node: $host"
    srun --nodes=1 --ntasks=1 --nodelist="$host" \
         ~/Reasoning360/scripts/tools/check_gpu.sh &
    pids["$host"]=$!
done

# Now wait for each job to finish and capture errors
error_found=0
for host in "${nodes[@]}"; do
    # wait returns the exit code of the process
    if ! wait "${pids[$host]}"; then
        echo "ERROR: Found GPU usage by other users on $host. Exiting."
        error_found=1
    fi
done

if [[ $error_found -eq 1 ]]; then
    exit 1
fi

echo "=== No leftover GPU usage found on all allocated nodes. ==="
echo "Proceeding with the main job..."


export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
# export GLOO_SOCKET_IFNAME=ens10f0np0



# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    /mnt/weka/home/zhuojun.cheng/miniconda3/envs/Reasoning360/bin/ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
        /mnt/weka/home/zhuojun.cheng/miniconda3/envs/Reasoning360/bin/ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10


# =================== leaderboard eval Config ===================
leaderboard_list=(
    "aime"
    "aime2025"
    # "barc"
    "finqa"
    "cruxeval-i"
    "cruxeval-o"
    "arcagi1"
)

n_nodes=8
n_gpus_per_node=8
gpu_ids=0,1,2,3,4,5,6,7

model_path=sail/Qwen2.5-Math-7B-Oat-Zero
data_folder=./data/test/
save_folder=./data/test_leaderboard_output/

# generation hyper-parameters
n_samples=1
batch_size=1024
temperature=1.0
top_k=-1 # 0 for hf rollout, -1 for vllm rollout
top_p=0.95
prompt_length=1024
response_length=3072
tensor_model_parallel_size=2
gpu_memory_utilization=0.8

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
domain_mappings["aime2025"]="math"
domain_mappings["math"]="math"
domain_mappings["minerva"]="math"
domain_mappings["olympiad_bench"]="math"
domain_mappings["gpqa"]="stem"
domain_mappings["arcagi1"]="simulation"
domain_mappings["barc"]="simulation"
domain_mappings["finqa"]="table"
domain_mappings["cruxeval-i"]="simulation"
domain_mappings["cruxeval-o"]="simulation"
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
        /mnt/weka/home/zhuojun.cheng/miniconda3/envs/Reasoning360/bin/python3 -m verl.trainer.main_generation \
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
            rollout.max_num_batched_tokens=4096 \
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