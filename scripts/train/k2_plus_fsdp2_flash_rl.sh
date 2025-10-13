#!/bin/bash
#SBATCH --job-name=flash_rl_tip_1_fp8_k2_plus_fsdp2
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --output=slurm/%x-%j.log
#SBATCH --error=slurm/%x-%j.log
#SBATCH --exclusive
#SBATCH --time=720:00:00
#SBATCH --partition=main

# =================== Conda Environment ===================
export CONDA_BIN_PATH=/lustrefs/users/varad.pimpalkhute/anaconda3/envs/sync-rl-v3/bin/

# =================== Frequently Used Variables ===================
RESUME_CKPT_DIR_NAME=""  # Fill in the checkpoint directory name to resume from, otherwise from scratch

# Fill in the llm-as-judge hosted URL, currently used only in 'STEM' domain
IDX=0
TIP_IMP_RATIO_CAP=(1.0 2.0 3.0 4.0 7.0 8.0 9.0 10.0)
NODE_NAME=(135 015 133 134 135 136 139 266)
export STEM_LLM_JUDGE_URL="http://azure-uk-hpc-H200-instance-${NODE_NAME[IDX]}:8000"
echo "STEM_LLM_JUDGE_URL: ${STEM_LLM_JUDGE_URL}"

export FLASHRL_LOGGING_LEVEL=DEBUG
export FLASHRL_CONFIG='bf16'

# =================== Cluster Environment ===================
export ROCR_VISIBLE_DEVICES=None
export NCCL_TIMEOUT_SECONDS=4800
export OMPI_MCA_coll_hcoll_enable=0 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
TORCH_NCCL_ENABLE_MONITORING=0 \
NCCL_SOCKET_IFNAME=eth0 \
UCX_TLS=rc \
UCX_NET_DEVICES=mlx5_ib0:1 \
NCCL_DEBUG=WARN \
NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
NCCL_IB_PCI_RELAXED_ORDERING=1 \
NCCL_IB_QPS_PER_CONNECTION=4 \
NCCL_IGNORE_CPU_AFFINITY=1 \
NCCL_P2P_NET_CHUNKSIZE=$((512 * 1024)) \
NCCL_PXN_DISABLE=1 \
NCCL_MIN_NCHANNELS=32 \
SHARP_SMX_UCX_INTERFACE=mlx5_ib0:1 \
SHARP_COLL_ENABLE_SAT=1 \
SHARP_COLL_LOG_LEVEL=3 \
SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1 \
NCCL_COLLNET_ENABLE=1

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

# =================== Data Mixture ===================
SHARED_DATA_PATH=/lustrefs/users/zhuojun.cheng/vpim/guru_data/
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/postprocessed_dedup_am_semantic_filtered_0.05_0.94_thresh_ratio0.5_sample1.0_balanced_step2
TEST_DATA_DIR=${SHARED_DATA_PATH}/test/online/

# Math (train)
math_train_path1=${TRAIN_DATA_DIR}/math__combined_118.2k.part1_scored.parquet
math_train_path2=${TRAIN_DATA_DIR}/math__combined_118.2k.part2_scored.parquet
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
aime25_test_path2=${TEST_DATA_DIR}/math__aime2025_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

# Code (train)
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__deduped_leetcode2k_2.4k_scored.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__deduped_livecodebench_599_scored.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__deduped_primeintellect_9.6k_scored.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__deduped_taco_11.1k_scored.parquet
# Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_500_sampled_200.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet

# Logic (train)
# arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111.parquet
# arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190.parquet
# barc_train_path=${TRAIN_DATA_DIR}/logic__barc_1.6k.parquet
# graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_1.2k.parquet
# ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k.parquet
# zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_1.3k.parquet
# Logic (test)
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_300_sampled_200.parquet
graph_test_path=${TEST_DATA_DIR}/logic__graph_logical_dataset_150_sampled_77.parquet
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_150_sampled_100.parquet
arcagi1_test_path=${TEST_DATA_DIR}/simulation__arcagi1_200.parquet

# Simulation (train)
# codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_3.7k.parquet
# Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_500_sampled_200.parquet

# Table (train)
# hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_4.3k.parquet
# multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_1.5k.parquet
# Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_300_sampled_200.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_300_sampled_200.parquet

# Stem (train)
# webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k.parquet
# Stem (test)
gpqa_diamond_test_path=${TEST_DATA_DIR}/stem__gpqa_diamond_198.parquet
supergpqa_test_path=${TEST_DATA_DIR}/stem__supergpqa_200.parquet

# Instruction follow (train)
if_train_path=${TRAIN_DATA_DIR}/ifbench_train_fixed_scored.parquet


train_files="['${math_train_path1}','${math_train_path2}','${leetcode_train_path}','${livecodebench_train_path}','${primeintellect_train_path}','${taco_train_path}','${if_train_path}']"
test_files="['${math_test_path}','${aime_test_path}','${aime25_test_path2}','${amc_test_path}','${humaneval_test_path}','${mbpp_test_path}','${livecodebench_test_path}','${gpqa_diamond_test_path}','${supergpqa_test_path}']"


# =================== Model ===================
BASE_MODEL=/lustrefs/users/runner/workspace/checkpoints/huggingface/sft/mid4_sft_reasoning_am/checkpoints/checkpoint_0001500

# =================== Logging ===================
WANDB_PROJECT=FlashRL360
WANDB_EXPERIMENT_NAME=${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${BASE_MODEL##*/}

# If RESUME_CKPT_DIR is not empty, resume from the checkpoint
if [[ -n "$RESUME_CKPT_DIR_NAME" ]]; then
    WANDB_EXPERIMENT_NAME="$RESUME_CKPT_DIR_NAME"
fi


# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ${CONDA_BIN_PATH}ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL \
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
    FLASHRL_LOGGING_LEVEL=DEBUG FLASHRL_CONFIG='bf16' \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
        FLASHRL_LOGGING_LEVEL=DEBUG FLASHRL_CONFIG='bf16' \
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
train_prompt_mini_bsz=32  # model grad update batchsize

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Training config
sp_size=16  # Reduced from 32 to reduce memory pressure
gen_tp=4
gen_max_num_seqs=50  # Reduced from 1024 to reduce memory pressure
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
rollout_max_num_batched_tokens=$(( (max_prompt_length + max_response_length) * 2))
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up modelforward, but note memory overflow
offload=True

# =================== Start RL training ===================

# Flash RL
# actor_rollout_ref.actor.tis_imp_ratio_cap=1.0
# actor_rollout_ref.rollout.calculate_log_probs=True
tip_imp_ratio_cap=${TIP_IMP_RATIO_CAP[IDX]}
calculate_log_probs=True

"${CONDA_BIN_PATH}python" -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name="dapo_fsdp_config.yaml" \
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
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT_SECONDS} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.actor.tis_imp_ratio_cap=${tip_imp_ratio_cap} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
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
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${gen_max_num_seqs} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs} \
    actor_rollout_ref.rollout.mode="sync" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    reward_model.reward_manager=async_multi_process \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$worker_num \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.log_val_generations=50 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1