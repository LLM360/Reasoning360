#!/bin/bash
#SBATCH --job-name=reasoning
#SBATCH --partition=mbzuai
#SBATCH --exclude=g42-h100-instance-[089-092,129-132,155-156]
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1024G
#SBATCH --output=slurm/verl_ppo_%j.out
#SBATCH --error=slurm/verl_ppo_%j.err
#SBATCH --exclusive
#SBATCH --time=3-12:00:00

export worker_num=$SLURM_NNODES

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=ens10f0np0
# export HYDRA_FULL_ERROR=1

# Experiment config
WORKING_DIR=${HOME}/Reasoning360
DATA_DIR=${WORKING_DIR}/data
orz_train_path=${DATA_DIR}/bigmath_preview_r1_after_sft/train.parquet
aime_test_path=${DATA_DIR}/bigmath_preview_r1_after_sft/aime_repeated_8x.parquet
amc_test_path=${DATA_DIR}/bigmath_preview_r1_after_sft/amc_repeated_4x.parquet
math_test_path=${DATA_DIR}/bigmath_preview_r1_after_sft/math.parquet
train_files="['$orz_train_path']"
test_files="['$aime_test_path', '$amc_test_path', '$math_test_path']"
BASE_MODEL="/mbz/shared/chengqian.gao/Qwen2.5-32B-SFT-11K-CKP184-Mar19"

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=shibo-math-sft-hsdp-dynamic-bsz-bigmath-filter-qwen-32b-run2

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mbz/users/yuheng.zha/openmpi/openmpi-5.0.7/lib

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"

# test nccl
/mbz/users/yuheng.zha/openmpi/openmpi-5.0.7/bin/mpirun -np $total_gpus \
    -host $host_list \
    --oversubscribe \
    -x NCCL_DEBUG=INFO \
    -map-by slot \
    /mbz/users/yuheng.zha/nccl-tests/build/all_reduce_perf \
    -b 8 \
    -e 1024M \
    -f 2 \
    -g 1 \
    | tee nccl_test_logs/nccl_bandwidth_test_$(date +%Y%m%d_%H%M%S).log


# We'll track PIDs so we can wait on them and detect errors
declare -A pids

# Spawn each check in the background
for host in "${nodes[@]}"; do
    echo "Spawning GPU check on node: $host"
    srun --nodes=1 --ntasks=1 --nodelist="$host" \
         ~/Reasoning360/scripts/check_gpu.sh &
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


# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS,HYDRA_FULL_ERROR=1 \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

sleep 5

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS,HYDRA_FULL_ERROR=1 \
        ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
    sleep 5
done

# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
ROLLOUT_TP_SIZE=8
SP_DEGREE=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=31744 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=32 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_DEGREE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    +actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    +actor_rollout_ref.rollout.use_partial_rollout=True \
    +actor_rollout_ref.rollout.partial_rollout_len=1024 \
    actor_rollout_ref.rollout.n=64 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=False \
    trainer.resume_from_path=False \
    trainer.nnodes=$worker_num \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.total_epochs=5

    # actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    # critic.ulysses_sequence_parallel_size=2 \