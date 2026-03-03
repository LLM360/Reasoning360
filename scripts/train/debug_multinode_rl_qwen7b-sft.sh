#!/bin/bash
#SBATCH --job-name=example-multinode-rl-qwen7b-instruct
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --account=iq
#SBATCH --mem=512G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --exclusive
#SBATCH --time=720:00:00


# =================== Frequently Used Variables ===================
RESUME_CKPT_DIR_NAME=""  # Fill in the checkpoint directory name to resume from, otherwise from scratch
export STEM_LLM_JUDGE_URL="http://10.24.1.81:8000"  # Fill in the llm-as-judge hosted URL, currently used only in 'STEM' domain

# =================== Cluster Environment ===================
export NCCL_DEBUG=info
export NCCL_ALGO=NVLSTree
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export TRITON_HOME="/tmp/triton_cache"
# export SANDBOX_FUSION_SERVERS="fs-mbz-cpu-002"

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
export VLLM_USE_V1=0

# =================== Data Mixture ===================
SHARED_DATA_PATH=/mnt/sharefs/users/chengqian.gao/guru
TEST_DATA_DIR=${SHARED_DATA_PATH}/offline_eval

# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet

# All test files across all domains
test_files="['${math_test_path}', '${aime_test_path}']"

# Training Data Configuration
DATA_MIX_DIR="/lustrefs/users/varad.pimpalkhute/data/k2/final/data_mix_1"
train_file_list=()

# List of datasets to include (filename only)
# Comment out lines to exclude specific datasets
dataset_names=(
    "codegen__deduped_leetcode2k_2.4k.parquet"
    "codegen__deduped_livecodebench_599.parquet"
    "codegen__deduped_primeintellect_9.6k.parquet"
    "codegen__deduped_taco_11.1k.parquet"
    "ifbench__fixed_85.6k.parquet"
    "logic__arcagi1_297.parquet"
    "logic__arcagi2_653.parquet"
    "logic__barc_3.4k.parquet"
    "logic__graph_logical_dataset_1.4k.parquet"
    "logic__ordering_puzzle_dataset_2.9k.parquet"
    "logic__reasoning_gym_40.6k.parquet"
    "logic__synlogic_12.1k.parquet"
    "logic__zebra_puzzle_dataset_5.0k.parquet"
    "math__combined_118.2k.part1.parquet"
    "math__combined_118.2k.part2.parquet"
    "omni_math_4.43k.parquet"
    "simulation__codeio_fixed_12.1k.parquet"
    "stem__nemotron_13.3k.parquet"
    "stem__web_31.7k.parquet"
    "table__hitab_7.4k.parquet"
    "table__multihier_2.9k.parquet"
)

echo "Collecting training files from ${DATA_MIX_DIR}..."

# Search for each dataset in all subdirectories
for dataset in "${dataset_names[@]}"; do
    for subdir in "impossible_questions" "131k_context_questions" "main_questions"; do
        file_path="${DATA_MIX_DIR}/${subdir}/${dataset}"
        if [ -f "$file_path" ]; then
            echo "Adding: $file_path"
            train_file_list+=("'$file_path'")
        fi
    done
done

# Join with comma to form Python list string
IFS=,
train_files="[${train_file_list[*]}]"
unset IFS

echo "Total training files found: ${#train_file_list[@]}"

# =================== Model ===================
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  # Note: This is the original Qwen32B-Base model. In training, we add 'think' system prompt to it (see README).
BASE_MODEL=/mnt/sharefs/users/haonan.li/models/Qwen2.5-7B-Instruct  # Note: This is the original Qwen32B-Base model. In training, we add 'think' system prompt to it (see README).

# =================== Logging ===================
WANDB_PROJECT=Difficulty-Aware-RL
WANDB_EXPERIMENT_NAME=${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${BASE_MODEL##*/}

# If RESUME_CKPT_DIR is not empty, resume from the checkpoint
if [[ -n "$RESUME_CKPT_DIR_NAME" ]]; then
    WANDB_EXPERIMENT_NAME="$RESUME_CKPT_DIR_NAME"
fi


# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10


# =================== RL Config ===================
# Note, we borrowed the config format from DAPO while here disabled all DAPO features to run the naive RL baseline.

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=256  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=16
train_prompt_mini_bsz=32  # model grad update batchsize

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
sp_size=1
gen_tp=1
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up modelforward, but note memory overflow
offload=True

# =================== Start RL training ===================
"${CONDA_BIN_PATH}python" -m verl.recipe.dapo.src.main_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.rollout.multi_turn.enable=False \
    +actor_rollout_ref.rollout.mode="sync" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=async_dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.nnodes=$worker_num \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=5 \
    +trainer.val_generations_to_log_to_wandb=30 \
    trainer.resume_mode=auto 