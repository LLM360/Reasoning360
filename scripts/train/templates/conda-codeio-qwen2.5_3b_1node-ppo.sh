# !/bin/bash

set -x

WORKING_DIR=$(pwd)
DATA_DIR=${WORKING_DIR}/data
train_path=${DATA_DIR}/codeio/train.parquet
test_path=${DATA_DIR}/codeio/test.parquet
BASE_MODEL=${WORKING_DIR}/models/Qwen2.5-3B

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=codeI/O-${BASE_MODEL##*/}-${SLURM_JOB_ID}

export WANDB_API_KEY=6eb16696cec55f88c62b7bbc82a5d16284c915cf
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=ens10f0np0

python -m verl.trainer.main_ppo \
    data.train_files=$train_path \
    data.val_files=$train_path \
    data.train_batch_size=1024 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${BASE_MODEL} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@
