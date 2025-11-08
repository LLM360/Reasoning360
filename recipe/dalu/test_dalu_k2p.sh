#!/bin/bash
#SBATCH --job-name=test-dalu-k2p-sft-dapo
#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --exclusive
#SBATCH --time=720:00:00


# =================== Frequently Used Variables ===================
RESUME_CKPT_DIR_NAME=""  # Fill in the checkpoint directory name to resume from, otherwise from scratch
export STEM_LLM_JUDGE_URL="http://azure-uk-hpc-H200-instance-150:8000"  # Fill in the llm-as-judge hosted URL, currently used only in 'STEM' domain

# =================== Cluster Environment ===================
# force IB and pick the rails explicitly
export OMPI_MCA_coll_hcoll_enable=0 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
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
    NCCL_COLLNET_ENABLE=1 \
    NCCL_TIMEOUT=7200 \
    NCCL_BLOCKING_WAIT=1 \
    TORCH_NCCL_TRACE_BUFFER_SIZE=1000 


export TRITON_HOME=/tmp/triton_cache

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
#TRAIN_DATA_DIR=/mnt/sharefs/users/zhuojun.cheng/guru_data/train/postprocessed_dedup_am
TRAIN_DATA_DIR=/lustrefs/users/haonan.li/data/k2/train_scored_dedup_am_12k_len_rm_flipscore_score_method_5_1_datamix_6
TEST_DATA_DIR=/lustrefs/users/haonan.li/data/k2/test_12k_len
# Math (train)
math_train1_path=${TRAIN_DATA_DIR}/math__combined_118.2k.part1.parquet
math_train2_path=${TRAIN_DATA_DIR}/math__combined_118.2k.part2.parquet
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime25_test_path=${TEST_DATA_DIR}/math__aime2025_repeated_8x_240.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

# Code (train)
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__deduped_leetcode2k_2.4k.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__deduped_livecodebench_599.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__deduped_primeintellect_9.6k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__deduped_taco_11.1k.parquet
# Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_500.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet

# Logic (train)
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_297.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_653.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_3.4k.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_dataset_1.4k.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_dataset_2.9k.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_dataset_5.0k.parquet
reasoning_gym_train_path=${TRAIN_DATA_DIR}/logic__reasoning_gym_40.6k.parquet
synlogic_train_path=${TRAIN_DATA_DIR}/logic__synlogic_12.1k.parquet

# Logic (test)
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet
reasoning_gym_test_path=${TEST_DATA_DIR}/logic__reasoning_gym_425.parquet
reasoning_gym_large_test_path=${TEST_DATA_DIR}/logic__reasoning_gym_4.3k.parquet
synlogic_test_path=${TEST_DATA_DIR}/logic__synlogic_217.parquet
arcagi1_test_path=${TEST_DATA_DIR}/logic__arcagi1_400.parquet

# Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_fixed_12.1k.parquet
# Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_200.parquet
cruxeval_i_test_path=${TEST_DATA_DIR}/simulation__cruxeval-i_800.parquet
cruxeval_o_test_path=${TEST_DATA_DIR}/simulation__cruxeval-o_800.parquet

# Table (train)
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_7.4k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_2.9k.parquet
# Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_336.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_1k.parquet
finqa_test_path=${TEST_DATA_DIR}/table__finqa_1.1k.parquet

# Stem (train)
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_31.7k.parquet
nemotron_train_path=${TRAIN_DATA_DIR}/stem__nemotron_13.3k.parquet
# Stem (test)
nemotron_test_path=${TEST_DATA_DIR}/stem__nemotron_100.parquet
nemotron_large_test_path=${TEST_DATA_DIR}/stem__nemotron_10.0k.parquet

gpqa_diamond_test_path=${TEST_DATA_DIR}/stem__gpqa_diamond_198.parquet
supergpqa_test_path=${TEST_DATA_DIR}/stem__supergpqa_1k.parquet

# IfBench (train)
ifbench_train_path=${TRAIN_DATA_DIR}/ifbench__fixed_85.6k.parquet # There might be bug, wait for fix
# IfBench (test)
ifbench_test_path=${TEST_DATA_DIR}/ifbench_800.parquet
ifbench_large_test_path=${TEST_DATA_DIR}/ifbench_8k.parquet

# OOD (test)
ifeval_test_path=${TEST_DATA_DIR}/ood__ifeval_100.parquet
livebench_data_analysis_test_path=${TEST_DATA_DIR}/ood__livebench_data_analysis_150.parquet
livebench_language_test_path=${TEST_DATA_DIR}/ood__livebench_language_140.parquet
livebench_reasoning_test_path=${TEST_DATA_DIR}/ood__livebench_reasoning_150.parquet

# Focused data mixture (math, code, stem)
train_files="['${math_train1_path}','${math_train2_path}','${leetcode_train_path}','${livecodebench_train_path}','${primeintellect_train_path}','${taco_train_path}','${webinstruct_train_path}','${nemotron_train_path}']"
test_files="['${math_test_path}','${aime_test_path}','${aime25_test_path}','${amc_test_path}','${humaneval_test_path}','${mbpp_test_path}','${livecodebench_test_path}','${nemotron_test_path}','${gpqa_diamond_test_path}','${supergpqa_test_path}']"

# Full data mixture (uncomment to use)
# train_files="['${math_train1_path}', '${math_train2_path}', '${leetcode_train_path}', '${livecodebench_train_path}', '${primeintellect_train_path}', '${taco_train_path}', '${arcagi1_train_path}', '${arcagi2_train_path}', '${zebra_train_path}', '${reasoning_gym_train_path}',  '${webinstruct_train_path}']"  # Use math as example, add to more tasks as needed
# # test_files="['${math_train1_path}']"
# test_files="['${aime25_test_path}', '${amc_test_path}', '${aime_test_path}', '${math_test_path}', '${humaneval_test_path}','${livecodebench_test_path}','${zebralogic_test_path}','${reasoning_gym_test_path}', '${nemotron_test_path}','${gpqa_diamond_test_path}','${ifeval_test_path}']"  # Use math as example, add to more tasks as needed test_files="['${supergpqa_test_path}','${ifeval_test_path}']"  # Use math as example, add to more tasks as needed

# =================== Model ===================
BASE_MODEL=/lustrefs/users/runner/workspace/checkpoints/huggingface/sft/mid4_sft_reasoning_am_cos_epoch/checkpoints/checkpoint_0002250
CONDA_BIN_PATH=/lustrefs/users/taylor.killian/miniconda3/envs/sync-rl/bin/

# =================== Logging ===================
WANDB_PROJECT=k2plus_rl
WANDB_EXPERIMENT_NAME=${SLURM_JOB_NAME}-${BASE_MODEL##*/}-${SLURM_JOB_ID}

# Set default local directory for checkpoints
DEFAULT_LOCAL_DIR="checkpoints/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME}"

# If RESUME_CKPT_DIR is not empty, resume from the checkpoint
if [[ -n "$RESUME_CKPT_DIR_NAME" ]]; then
    WANDB_EXPERIMENT_NAME=$(basename "$RESUME_CKPT_DIR_NAME")
    DEFAULT_LOCAL_DIR="$RESUME_CKPT_DIR_NAME"
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
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL \
        env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES \
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
clip_ratio_high=0.28

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 60))
max_validation_length=$((1024 * 60))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=16
train_prompt_mini_bsz=32  # model grad update batchsize

# Algorithm
temperature=1.4
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Training config
sp_size=8
gen_tp=8
gen_max_num_seqs=1024
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 1))  # increase this to speed up modelforward, but note memory overflow
offload=True

# =================== Start RL training ===================
"${CONDA_BIN_PATH}python" -m recipe.dalu.main_dalu \
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
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp2" \
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
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    +actor_rollout_ref.rollout.validation_length=${max_validation_length} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
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
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.mode="sync" \
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
    trainer.total_epochs=10 \
    trainer.log_val_generations=1 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    +trainer.run_id=${WANDB_ID} \
    +trainer.enable_budget=True \
    +data.dynamic_filtering=True \
    +data.pass_rate_upper_bound=0.9 \
    +data.initial_pass_rate_column=deepseek_r1_0528_pass_rate
