#!/bin/bash
#SBATCH --job-name=zhoujun-trial-rl-codegen
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH --exclusive
#SBATCH --time=720:00:00
#SBATCH --qos=iq


# =================== Environment ===================
# may vary from cluster to cluster, please check the environment variables
# export LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:$LD_LIBRARY_PATH \
#        UCX_TLS=dc \
#        UCX_NET_DEVICES=mlx5_ib0:1 \
#        CUDA_DEVICE_ORDER=PCI_BUS_ID \
#        NCCL_SOCKET_IFNAME=eth0 \
#        NCCL_DEBUG=WARN \
#        NCCL_NET_GDR_LEVEL=5 \
#        NCCL_MIN_NCHANNELS=32 \
#        NCCL_TOPO_FILE=/mnt/users/runner/scripts/ndv5-topo.xml \
#        OMPI_MCA_coll_hcoll_enable=0 \
#        OMPI_MCA_plm_rsh_no_tree_spawn=1 \
#        OMPI_MCA_plm_rsh_num_concurrent=800 \
#        NCCL_IB_QPS_PER_CONNECTION=4 \
#        NCCL_P2P_NET_CHUNKSIZE=$((512*1024)) \
#        NCCL_PXN_DISABLE=1

# export UCX_NET_DEVICES=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
# export CUDA_DEVICE_MAX_CONNECTIONS=1

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

# Spawn each check in the background
for host in "${nodes[@]}"; do
    echo "Spawning GPU check on node: $host"
    srun --nodes=1 --ntasks=1 --nodelist="$host" \
         ~/Reasoning360/scripts/tools/check_gpu.sh &
    pids["$host"]=$!
done

# Now wait for each job to finish and capture errors
error_found=0
for host in "${nodes[@]}"; do
    # wait returns the exit code of the process
    if ! wait "${pids[$host]}"; then
        echo "ERROR: Found GPU usage by other users on $host. Exiting."
        error_found=1
    fi
done

if [[ $error_found -eq 1 ]]; then
    exit 1
fi

echo "=== No leftover GPU usage found on all allocated nodes. ==="
echo "Proceeding with the main job..."


export head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

export worker_num=$SLURM_NNODES
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
# export GLOO_SOCKET_IFNAME=ens10f0np0


# =================== Data Mixture (genie-25K)===================
# WORKING_DIR=${HOME}/Reasoning360
WORKING_DIR=/mnt/weka/home/zhuojun.cheng/leo/Reasoning360/
TRAIN_DATA_DIR=${WORKING_DIR}/data/train_raw
TEST_DATA_DIR=${WORKING_DIR}/data/test
# Math (train)
math_train_path=${TRAIN_DATA_DIR}/math__patch_merged_deduped_13.3k.parquet
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet
# Code (train)
leetcode2k_train_path=${TRAIN_DATA_DIR}/codegen__deduped_leetcode2k_2.4k.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__deduped_livecodebench_599.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__deduped_primeintellect_9.6k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__deduped_taco_11.1k.parquet
# Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_500.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet
# Logic (train)
zebralogic_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_dataset_5.7k.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_dataset_2.8k.parquet
ordering_puzzle_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_dataset_2.9k.parquet
# Logic (test)
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_300.parquet
graph_test_path=${TEST_DATA_DIR}/logic__graph_logical_dataset_150.parquet
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_150.parquet
# Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_12.1k.parquet  # TODO
# Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_500.parquet
cruxeval_o_test_path=${TEST_DATA_DIR}/simulation__cruxeval-o_800.parquet
cruxeval_i_test_path=${TEST_DATA_DIR}/simulation__cruxeval-i_800.parquet
# Table (train)
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_2.9k.parquet
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_7.4k.parquet
# Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_300.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_300.parquet
# Stem (train)
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__33.0k.parquet # TODO
# Stem (test)
gpqa_diamond_test_path=${TEST_DATA_DIR}/stem__gpqa_198.parquet
# ARC-AGI (train)
arc_agi_train_path=${TRAIN_DATA_DIR}/simulation__arcagi1_297.parquet
# ARC-AGI (test)
arc_agi_test_path=${TEST_DATA_DIR}/simulation__arcagi1_200.parquet


train_files="['${leetcode2k_train_path}', '${primeintellect_train_path}', '${taco_train_path}', '${livecodebench_train_path}']"

test_files="['${math_test_path}',\
'${aime_test_path}',\
'${amc_test_path}',\
'${humaneval_test_path}',\
'${mbpp_test_path}',\
'${livecodebench_test_path}',\
'${zebralogic_test_path}',\
'${graph_test_path}',\
'${ordering_puzzle_test_path}',\
'${codeio_test_path}',\
'${multihier_test_path}',\
'${hitab_test_path}',\
'${gpqa_diamond_test_path}',\
'${arc_agi_test_path}']"


# =================== Model ===================
LOCAL_MODEL_DIR=${HOME}/.cache/huggingface/hub
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
# BASE_MODEL=Qwen/Qwen2.5-3B
# BASE_MODEL=${LOCAL_MODEL_DIR}/models--Qwen--Qwen2.5-7B-think
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# =================== Logging ===================
WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=${SLURM_JOB_ID}-${SLURM_JOB_NAME}-${BASE_MODEL##*/}


# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &    
done
sleep 10


# =================== RL Config ===================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512  # grad accum bsz; real grad accum bsz: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))  # rollout bsz, i.e., the x-axis in RL plot
n_resp_per_prompt=16
train_prompt_mini_bsz=32

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
sp_size=1
gen_tp=2
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
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
    data.truncation='left' \
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
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
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
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    +trainer.val_generations_to_log_to_wandb=50 \
    trainer.resume_mode=auto