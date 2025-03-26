#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --job-name=rl
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm/verl-%j.out
#SBATCH --error=slurm/verl-%j.err
#SBATCH --exclude=g42-odin-h100-[106,342,358,396,496]
#SBATCH --exclusive


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=ens10f0np0
export worker_num=$SLURM_NNODES
export PYTHONPATH=/Reasoning360:$PYTHONPATH

WORKING_DIR=${HOME}/Reasoning360
MOUNT_WORKING_DIR=/Reasoning360
IMAGE_PATH=${HOME}/Reasoning360/docker/images/verl_megatron_v2.sqsh

# Data config
DATA_DIR=${MOUNT_WORKING_DIR}/data
deepscaler_train_path=${DATA_DIR}/deepscaler_preview/train.parquet
aime_test_path=${DATA_DIR}/deepscaler_preview/aime.parquet
amc_test_path=${DATA_DIR}/deepscaler_preview/amc.parquet
math_test_path=${DATA_DIR}/deepscaler_preview/math.parquet
minerva_test_path=${DATA_DIR}/deepscaler_preview/minerva.parquet
olympiad_bench_test_path=${DATA_DIR}/deepscaler_preview/olympiad_bench.parquet

train_files="[${deepscaler_train_path}]"
test_files="[${aime_test_path},${amc_test_path},${math_test_path},${minerva_test_path},${olympiad_bench_test_path}]"

# Model config
# BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
BASE_MODEL=Qwen/Qwen2.5-32B
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# BASE_MODEL=meta-llama/Llama-3.3-70B-Instruct
# BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

# Parallel config
SP_SIZE=2
ROLLOUT_TP_SIZE=8

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=zhoujun-docker-math-${BASE_MODEL##*/}

echo "Node list: ${nodes[@]}"

srun --container-image=$IMAGE_PATH --container-mounts="${WORKING_DIR}:/Reasoning360" --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

echo "================================================"
echo "Starting HEAD at $head_node"
srun --container-image=$IMAGE_PATH --container-mounts="${WORKING_DIR}:/Reasoning360" --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus 8 --block &
sleep 30
echo "RAY HEAD STARTED!"

echo "================================================"
for ((i = 1; i < worker_num - 1; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --container-image=$IMAGE_PATH --container-mounts="${WORKING_DIR}:/Reasoning360" --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        ray start --address "$address_head" --num-cpus $SLURM_CPUS_PER_TASK --num-gpus 8 --block &
    sleep 10
    echo "RAY WORKER $i STARTED!"
done


cmd="python3 /Reasoning360/verl/trainer/main_ppo.py  --config-path=/Reasoning360/verl/trainer/config --config-name='ppo_trainer' \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.n=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$worker_num \
    trainer.save_freq=100 \
    trainer.test_freq=3 \
    trainer.total_epochs=5 \
    trainer.val_generations_to_log_to_wandb=30"

node_i=${nodes[worker_num - 1]}
echo "================================================"
echo "Launching job at node $node_i..."
full_cmd="ray start --address "$address_head" \
    --num-cpus $SLURM_CPUS_PER_TASK --num-gpus 8 && \
    sleep 20 && $cmd"
srun --overlap --container-image=$IMAGE_PATH --container-mounts="${WORKING_DIR}:/Reasoning360" --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    ray status
srun --overlap --container-image=$IMAGE_PATH --container-mounts="${WORKING_DIR}:/Reasoning360" --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
    bash -c "cd /Reasoning360/ && pip install -e . && $full_cmd" 


