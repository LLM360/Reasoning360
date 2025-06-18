#!/bin/bash
#SBATCH --job-name=server_llm_2node
#SBATCH --account=iq
#SBATCH --nodes=2 
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=slurm/serve_llm_as_verifier_%j.log
#SBATCH --error=slurm/serve_llm_as_verifier_%j.err

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Error: Model name is required"
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME=$1
echo "Using model: $MODEL_NAME"

# =================== Cluster Environment ===================
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
         bash ~/Reasoning360/scripts/tools/check_gpu.sh &
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

unset LD_LIBRARY_PATH

# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    /mnt/weka/home/haonan.li/miniconda3/envs/verl/bin/ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
        /mnt/weka/home/haonan.li/miniconda3/envs/verl/bin/ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10

# (2) export judge URL for downstream clients
export STEM_LLM_JUDGE_URL="http://${head_node_ip}:8000"
echo "STEM_LLM_JUDGE_URL=$STEM_LLM_JUDGE_URL"

# (3) Set up Ray cluster environment variables
export VLLM_HOST_IP=$head_node_ip
export CONDA_BIN_PATH=/mnt/weka/home/haonan.li/miniconda3/envs/verl/bin/


# (5) Only run vLLM server on the head node
if [ "$SLURM_NODEID" -eq 0 ]; then
    # Calculate tensor and pipeline parallel sizes
    # Assuming 8 GPUs per node, using all GPUs for tensor parallelism
    # and using pipeline parallelism across nodes
    TOTAL_GPUS=$((SLURM_NNODES * 8))
    TENSOR_PARALLEL_SIZE=8  # Use all GPUs in each node for tensor parallelism
    PIPELINE_PARALLEL_SIZE=$SLURM_NNODES  # Use all nodes for pipeline parallelism

    echo "Starting vLLM server with tensor_parallel_size=$TENSOR_PARALLEL_SIZE and pipeline_parallel_size=$PIPELINE_PARALLEL_SIZE"
    ${CONDA_BIN_PATH}vllm serve "$MODEL_NAME" \
        --host "$head_node_ip" \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --pipeline-parallel-size $PIPELINE_PARALLEL_SIZE \
        --max_model_len 32000
fi
# test the server
# curl http://$NODE_IP:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "Qwen/Qwen3-32B",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Write a one-sentence bedtime story about a unicorn."
#             }
#         ]
#     }'
