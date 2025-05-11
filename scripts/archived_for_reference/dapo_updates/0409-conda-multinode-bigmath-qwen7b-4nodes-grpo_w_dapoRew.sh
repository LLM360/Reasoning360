#!/bin/bash
#SBATCH --job-name=rl-grpo+
#SBATCH --partition=mbzuai
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1T
#SBATCH --output=slurm/verl-grpo+-%j.out
#SBATCH --error=slurm/verl-grpo+-%j.err
#SBATCH --exclusive
#SBATCH --time=720:00:00

# sleep 100
#!/bin/bash

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"

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


export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# 176.56.202.149
# address_head=176.56.202.149:6379

# Experiment config
WORKING_DIR=${HOME}/Reasoning360

DATA_DIR=${WORKING_DIR}/data
# math_train_path=${DATA_DIR}/math/train.parquet
orz_train_path=${DATA_DIR}/bigmath_preview/train.parquet
# deepscaler_train_path=${DATA_DIR}/deepscaler_preview/train.parquet
# math_test_path=${DATA_DIR}/math/test.parquet
aime_test_path=${DATA_DIR}/bigmath_preview/aime_repeated_8x.parquet
amc_test_path=${DATA_DIR}/bigmath_preview/amc_repeated_4x.parquet
math_test_path=${DATA_DIR}/bigmath_preview/math.parquet
# minerva_test_path=${DATA_DIR}/deepscaler_preview/minerva.parquet
# olympiad_bench_test_path=${DATA_DIR}/deepscaler_preview/olympiad_bench.parquet

train_files="['${orz_train_path}']"
test_files="['${aime_test_path}', '${amc_test_path}', '${math_test_path}']" #, '$minerva_test_path', '$olympiad_bench_test_path']"

# Model config
# BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
BASE_MODEL=Qwen/Qwen2.5-Math-7B
# BASE_MODEL=Qwen/Qwen2.5-32B

# Parallel config
SP_SIZE=1
ROLLOUT_TP_SIZE=4

# Wandb config
WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=taylor-7B-bigmath-grpo_w_dapoRew_rewardMetricTest-${BASE_MODEL##*/}-${SLURM_JOB_ID}

export worker_num=$SLURM_NNODES
# export worker_num=4
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export GLOO_SOCKET_IFNAME=ens10f0np0
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster


# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# ray start --head --node-ip-address=176.56.202.149 --port=6379 --num-cpus 96 --num-gpus 8


# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10

# address_head=176.56.202.149:6379
# ray start --address $address_head --num-cpus 96  --num-gpus 8


# Start training
"${CONDA_BIN_PATH}python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=2048 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.n=64 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward_model.reward_manager=dapo \
    reward_model.reward_metric=dapo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$worker_num \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.val_generations_to_log_to_wandb=10