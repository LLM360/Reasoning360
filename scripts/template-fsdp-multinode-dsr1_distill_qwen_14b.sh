#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --partition=mbzuai
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/verl-%j.out
#SBATCH --error=slurm/verl-%j.err
#SBATCH --exclusive
#SBATCH --time=12:00:00

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# Experiment config
WORKING_DIR=${HOME}/Reasoning360
DATA_DIR=${WORKING_DIR}/data
math_train_path=${DATA_DIR}/math/train.parquet
math_test_path=${DATA_DIR}/math/test.parquet
train_files="['$math_train_path']"
test_files="['$math_test_path']"
BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=math-${BASE_MODEL##*/}-${SLURM_JOB_ID}

export worker_num=$SLURM_NNODES
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=ens10f0np0

# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
    sleep 10
done

# Start training
"${CONDA_BIN_PATH}python" -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.val_batch_size=1312 \
    data.max_prompt_length=4096 \
    data.max_response_length=10240 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.fsdp_size=-1 \
    critic.model.fsdp_config.grad_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.path=$BASE_MODEL \
    critic.model.use_remove_padding=True \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=32 \
    critic.ulysses_sequence_parallel_size=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=False \
    trainer.nnodes=$worker_num \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=10