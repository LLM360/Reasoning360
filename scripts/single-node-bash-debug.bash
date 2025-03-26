# a minimal script to run a single node PPO training
# This script is used to debug any code updates
# It is not intended to be used for training

export head_node=${nodes[0]}

head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# Experiment config
WORKING_DIR=${HOME}/Reasoning360
DATA_DIR=${WORKING_DIR}/data
deepscaler_train_path=${DATA_DIR}/orz_zero_style_v2/train.parquet
aime_test_path=${DATA_DIR}/orz_zero_style_v2/aime.parquet

train_files="['$deepscaler_train_path']"
test_files="['$aime_test_path']"
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=orz-zero-debug9-${BASE_MODEL##*/}-${SLURM_JOB_ID}

export worker_num=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=ens10f0np0
export HYDRA_FULL_ERROR=1

# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 96 --num-gpus 8

# Start training
"${CONDA_BIN_PATH}python" -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=16 \
    data.val_batch_size=1312 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.fsdp_config.fsdp_size=-1 \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.path=$BASE_MODEL \
    critic.model.use_remove_padding=True \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.ppo_mini_batch_size=16 \
    critic.ulysses_sequence_parallel_size=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=True \
    trainer.nnodes=$worker_num \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.val_generations_to_log_to_wandb=30 \
    reward_model.reward_manager=llm_judge \
    reward_model.reward_metric=math_llm_judge