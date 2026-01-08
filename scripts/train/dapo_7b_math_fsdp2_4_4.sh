#!/bin/bash
#SBATCH --job-name=dapo-7b-math-fsdp2-4-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=slurm/%x-%j.log
#SBATCH --error=slurm/%x-%j.log
#SBATCH --exclusive
#SBATCH --time=720:00:00
#SBATCH --partition=main

# NOTE: added by Reasoning360.
export MATH_LLM_JUDGE_URL=http://azure-uk-hpc-H200-instance-033:8000

export CONDA_BIN_PATH=/mnt/weka/home/varad.pimpalkhute/anaconda3/envs/sync-rl-v1/bin/
export ROCR_VISIBLE_DEVICES=None 
export NCCL_TIMEOUT_SECONDS=4800
export NCCL_DEBUG=warn
export NCCL_NET=IB
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
export NCCL_CROSS_NIC=1
export NCCL_IB_TC=136
export NCCL_SOCKET_IFNAME="^lo,docker,virbr"
export CUDA_DEVICE_MAX_CONNECTIONS=8
export NCCL_NVLS_ENABLE=1

project_name='DAPO'
exp_name='DAPO-Qwen2.5-7b-MATH-0527a1-fsdp2-fully-async-4-4'

export VERL_USE_THREAD_TIMEOUT=false

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# Paths
# RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl_old"}
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-Math-7B"}
# CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}


# TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}

# Training Data Configuration
DATA_MIX_DIR="/mnt/weka/home/varad.pimpalkhute/data/k2/final/data_mix_1"
train_file_list=()
test_file_list_impossible_questions=()

# List of datasets to include (filename only)
# Comment out lines to exclude specific datasets
dataset_names=(
    # "codegen__deduped_leetcode2k_2.4k.parquet"
    # "codegen__deduped_livecodebench_599.parquet"
    # "codegen__deduped_primeintellect_9.6k.parquet"
    # "codegen__deduped_taco_11.1k.parquet"
    "ifbench__fixed_85.6k.parquet"
    # "logic__arcagi1_297.parquet"
    # "logic__arcagi2_653.parquet"
    # "logic__barc_3.4k.parquet"
    # "logic__graph_logical_dataset_1.4k.parquet"
    # "logic__ordering_puzzle_dataset_2.9k.parquet"
    # "logic__reasoning_gym_40.6k.parquet"
    # "logic__synlogic_12.1k.parquet"
    # "logic__zebra_puzzle_dataset_5.0k.parquet"
    # "math__combined_118.2k.part1.parquet"
    # "math__combined_118.2k.part2.parquet"
    # "omni_math_4.43k.parquet"
    # "simulation__codeio_fixed_12.1k.parquet"
    # "stem__nemotron_13.3k.parquet"
    # "stem__web_31.7k.parquet"
    # "table__hitab_7.4k.parquet"
    # "table__multihier_2.9k.parquet"
)

echo "Collecting training files from ${DATA_MIX_DIR}..."

# Search for each dataset in all subdirectories
# NOTE: added by Reasoning360. Exclude impossible_questions from training data
for dataset in "${dataset_names[@]}"; do
    for subdir in "131k_context_questions" "main_questions"; do
        file_path="${DATA_MIX_DIR}/${subdir}/${dataset}"
        if [ -f "$file_path" ]; then
            echo "Adding: $file_path"
            train_file_list+=("'$file_path'")
        fi
    done
done

for dataset in "${dataset_names[@]}"; do
    for subdir in "impossible_questions"; do
        file_path="${DATA_MIX_DIR}/${subdir}/${dataset}"
        if [ -f "$file_path" ]; then
            echo "Adding: $file_path"
            test_file_list_impossible_questions+=("'$file_path'")
        fi
    done
done

test_file_list_impossible_questions+=("'/mnt/weka/home/varad.pimpalkhute/async/Reasoning360/data/aime-2024.parquet'")

echo "Test files for impossible questions: ${#test_file_list_impossible_questions[@]}"
echo "Total training files found: ${#train_file_list[@]}"

# Join with comma to form Python list string
IFS=,
train_files="[${train_file_list[*]}]"
test_files="[${test_file_list_impossible_questions[*]}]"
unset IFS

echo "Test files for impossible questions: ${test_files}"
echo "Training files: ${train_files}"



# =================== Ray node setup ===================

# Get the list of allocated nodes
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "Nodes to check: ${nodes[@]}"

# We'll track PIDs so we can wait on them and detect errors
declare -A pids
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1

# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ${CONDA_BIN_PATH}ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES VERL_USE_THREAD_TIMEOUT=${VERL_USE_THREAD_TIMEOUT} \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES VERL_USE_THREAD_TIMEOUT=${VERL_USE_THREAD_TIMEOUT} \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10
# =================== Ray node setup end ===================

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 32))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
ref_offload=True
actor_offload=False
gen_tp=1
sp_size=4
fsdp_size=4

# Fully async specific parameters
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

n_gpus_rollout=4
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*100)))
test_freq=10
staleness_threshold=0.1
trigger_parameter_sync_step=4
require_batches=4
partial_rollout=True

MODEL_PATH="models/Qwen2.5-Math-7B"
CKPTS_DIR="ckpts/${project_name}/${exp_name}"

"${CONDA_BIN_PATH}python3" -m reasoning360.recipe.fully_async_policy.fully_async_main \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    reward_model.reward_manager=dapo \
    reward_model.launch_reward_fn_async=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=True  \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs=10 \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_rollout_log_probs=True