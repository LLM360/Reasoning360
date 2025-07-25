#!/bin/bash
#SBATCH --job-name=16node-32kgen-archer-qwen32b-amthink
#SBATCH --partition=main
#SBATCH --account=iq
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --exclusive
#SBATCH --time=720:00:00


# =================== Frequently Used Variables ===================
RESUME_CKPT_DIR_NAME=""  # Fill in the checkpoint directory name to resume from, otherwise from scratch
export STEM_LLM_JUDGE_URL="http://10.24.24.45:8000"  # Fill in the llm-as-judge hosted URL, currently used only in 'STEM' domain

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
export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# =================== Data Mixture ===================
SHARED_DATA_PATH=/mnt/sharefs/users/zhuojun.cheng
SHARED_MODEL_PATH=/mnt/sharefs/users/haonan.li/models
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/guru_data/train/guru92k_release_0603
TEST_DATA_DIR=${SHARED_DATA_PATH}/guru_data/test/online      # ← unchanged

# ---------- Math ----------
# train
math_train_path=${TRAIN_DATA_DIR}/math__combined_54.4k.parquet
# test
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

# ---------- Code ----------
# train
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__deduped_leetcode2k_1.3k.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__deduped_livecodebench_440.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__deduped_primeintellect_7.5k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__deduped_taco_8.8k.parquet
# test (unchanged)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_500_sampled_200.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet

# ---------- Logic ----------
# train
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_1.6k.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_dataset_1.2k.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_dataset_1.9k.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_dataset_1.3k.parquet
# test (unchanged)
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_300_sampled_200.parquet
graph_test_path=${TEST_DATA_DIR}/logic__graph_logical_dataset_150_sampled_77.parquet
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_150_sampled_100.parquet
arcagi1_test_path=${TEST_DATA_DIR}/simulation__arcagi1_200.parquet

# ---------- Simulation ----------
# train
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_fixed_12.1k_3.7k.parquet
# test (unchanged)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_500_sampled_200.parquet

# ---------- Table ----------
# train
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_4.3k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_1.5k.parquet
# test (unchanged)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_300_sampled_200.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_300_sampled_200.parquet

# ---------- Stem ----------
# train
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k.parquet
# test (unchanged)
gpqa_diamond_test_path=${TEST_DATA_DIR}/stem__gpqa_diamond_198.parquet
supergpqa_test_path=${TEST_DATA_DIR}/stem__supergpqa_200.parquet

train_files="['${math_train_path}', \
'${leetcode_train_path}', \
'${livecodebench_train_path}', \
'${primeintellect_train_path}', \
'${taco_train_path}', \
'${arcagi1_train_path}', \
'${arcagi2_train_path}', \
'${barc_train_path}', \
'${graph_train_path}', \
'${ordering_train_path}', \
'${zebra_train_path}', \
'${codeio_train_path}', \
'${hitab_train_path}', \
'${multihier_train_path}', \
'${webinstruct_train_path}']"

test_files="['${math_test_path}', \
'${aime_test_path}', \
'${amc_test_path}', \
'${humaneval_test_path}', \
'${mbpp_test_path}', \
'${livecodebench_test_path}', \
'${zebralogic_test_path}', \
'${graph_test_path}', \
'${ordering_puzzle_test_path}', \
'${arcagi1_test_path}', \
'${codeio_test_path}', \
'${multihier_test_path}', \
'${hitab_test_path}', \
'${gpqa_diamond_test_path}', \
'${supergpqa_test_path}']"

# =================== Model ===================
# BASE_MODEL=${SHARED_DATA_PATH}/Qwen2.5-32B-think  # Note: this is Qwen32B-Base model with 'think' system prompt
BASE_MODEL=${SHARED_MODEL_PATH}/32b-am-1084

# =================== Logging ===================
WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=${SLURM_JOB_ID}-${SLURM_JOB_NAME} #-${BASE_MODEL##*/}

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
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28
grad_clip=1.0

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 32))
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
train_prompt_mini_bsz=64  # model grad update batchsize

# Algorithm
train_temp=1.0
val_temp=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
sp_size=8
gen_tp=4
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up modelforward, but note memory overflow
offload=True

n_resp_per_prompt_val=1
total_epochs=10
save_freq=10
test_freq=10
max_ckpt_to_keep=2
enable_curriculum=True
val_before_train=True

# Token Filtering
use_token_entropy_separate=True
use_token_entropy_filter=False
token_entropy_quantile=0.8
high_entropy_kl_loss_scale_coef=0.0
low_entropy_clip_ratio_low=0.2
low_entropy_clip_ratio_high=0.2
high_entropy_clip_ratio_low=0.5
high_entropy_clip_ratio_high=0.5


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
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    +actor_rollout_ref.actor.use_token_entropy_separate=${use_token_entropy_separate} \
    +actor_rollout_ref.actor.use_token_entropy_filter=${use_token_entropy_filter} \
    +actor_rollout_ref.actor.token_entropy_quantile=${token_entropy_quantile} \
    +actor_rollout_ref.actor.high_entropy_kl_loss_scale_coef=${high_entropy_kl_loss_scale_coef} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_low=${low_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_high=${low_entropy_clip_ratio_high} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_low=${high_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_high=${high_entropy_clip_ratio_high} \
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
    actor_rollout_ref.rollout.temperature=${train_temp} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temp} \
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
    trainer.val_before_train=${val_before_train} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.nnodes=$worker_num \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    +trainer.val_generations_to_log_to_wandb=30 \
    trainer.resume_mode=auto