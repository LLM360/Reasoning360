#!/bin/bash
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-node=2         
#SBATCH --time=4-00:00:00
#SBATCH --job-name rl_training_llm360_ppo
#SBATCH --mem=128G 

set -x

WORKING_DIR=${HOME}/Reasoning360
DATA_DIR=${WORKING_DIR}/examples/data_preprocess/data
reasoning_train_path=${DATA_DIR}/graph_dataset/graph_dataset_train.parquet
reasoning_test_path=${DATA_DIR}/graph_dataset/graph_dataset_test.parquet
train_files="['$reasoning_train_path']"
test_files="['$reasoning_test_path']"
BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=graph_puzzle-${BASE_MODEL##*/}-${SLURM_JOB_ID}

export VLLM_ATTENTION_BACKEND=XFORMERS
# export GLOO_SOCKET_IFNAME=ens10f0np0

"${CONDA_BIN_PATH}python" -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.val_batch_size=6312 \
    data.max_prompt_length=100000 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.default_local_dir=/scratch/gilbreth/dey33/checkpoints/\${trainer.project_name}/$EXPERIMENT_NAME \
    trainer.total_epochs=15 $@
