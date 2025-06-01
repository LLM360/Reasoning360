#!/bin/bash
#SBATCH --job-name=32b-16node-guru-full-minibsz64
#SBATCH --partition=main
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH --exclusive
#SBATCH --time=192:00:00


# =================== Environment ===================
# may vary from cluster to cluster, please check the environment variables
# === M1
export LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:$LD_LIBRARY_PATH \
       UCX_TLS=dc \
       UCX_NET_DEVICES=mlx5_ib0:1 \
       CUDA_DEVICE_ORDER=PCI_BUS_ID \
       NCCL_SOCKET_IFNAME=eth0 \
       NCCL_DEBUG=WARN \
       NCCL_NET_GDR_LEVEL=5 \
       NCCL_MIN_NCHANNELS=32 \
       NCCL_TOPO_FILE=/mnt/users/runner/scripts/ndv5-topo.xml \
       OMPI_MCA_coll_hcoll_enable=0 \
       OMPI_MCA_plm_rsh_no_tree_spawn=1 \
       OMPI_MCA_plm_rsh_num_concurrent=800 \
       NCCL_IB_QPS_PER_CONNECTION=4 \
       NCCL_P2P_NET_CHUNKSIZE=$((512*1024)) \
       NCCL_PXN_DISABLE=1

export UCX_NET_DEVICES=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export STEM_LLM_JUDGE_URL=http://10.0.4.181:8000

# === M2 setup ===
# export NCCL_DEBUG=info
# export NCCL_ALGO=NVLSTree
# export NCCL_IBEXT_DISABLE=1
# export NCCL_NVLS_ENABLE=1
# export NCCL_IB_HCA=mlx5
# export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# REAL_DIR=~/Reasoning360/ray_logs/$SLURM_JOB_ID
# mkdir -p "$REAL_DIR"
# ln -sfn "$REAL_DIR" ~/r_$SLURM_JOB_ID

# LOG_ROOT=~/r_$SLURM_JOB_ID

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
export WANDB_API_KEY=633cdb1f1b9dfb2ae2681e47863635fe33b93a10
export CONDA_BIN_PATH=/lustrefs/users/shibo.hao/miniforge3/envs/Reasoning360-May/bin/
export HYDRA_FULL_ERROR=1
# export GLOO_SOCKET_IFNAME=ens10f0np0


# =================== Data Mixture (genie-25K)===================

BASE_MODEL=${HOME}/Qwen2.5-32B-think
WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=azure-hpc-H200-instance-010.core42.ai-20250512_013325-Qwen2.5-32B-think-16node-guru-full-minibsz64

WORKING_DIR=/lustrefs/users/zhuojun.cheng/Reasoning360
TRAIN_DATA_DIR=${WORKING_DIR}/data/train_guru_full
TEST_DATA_DIR=${WORKING_DIR}/data/test/test
# Math (train)
# math_train_path1=${TRAIN_DATA_DIR}/math__merged_deduped_l1e-5_h0.9_60.0k_sampled_60.0k.parquet
# math_train_path2=${TRAIN_DATA_DIR}/math__patch_merged_deduped_13.3k_l1e-5_h0.9_9.3k_sampled_9.3k.parquet
math_train_path=/lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered/math__40k.parquet  # abs path
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet
# Code (train-deprecated)
# leetcode_train_path=${TRAIN_DATA_DIR}/codegen__deduped_leetcode2k_2.4k_l1e-5_h0.9_1.3k_sampled_177.parquet  # TODO: change this
# livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__deduped_livecodebench_599_l1e-5_h0.9_451_sampled_61.parquet  # TODO: change this
# primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__deduped_primeintellect_9.6k_l1e-5_h0.9_7.6k_sampled_1.0k.parquet  # TODO: change this
# taco_train_path=${TRAIN_DATA_DIR}/codegen__deduped_taco_11.1k_l1e-5_h0.9_8.9k_sampled_1.2k.parquet  # TODO: change this
# Code (train)
leetcode_train_path=/lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered/codegen__deduped_leetcode2k_2.4k_l1e-5_h0.9_1.3k_reduced_tests.parquet  # abs path
livecodebench_train_path=/lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered/codegen__deduped_livecodebench_599_l1e-5_h0.9_451_reduced_tests.parquet  # abs path
primeintellect_train_path=/lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered/codegen__deduped_primeintellect_9.6k_l1e-5_h0.9_7.6k_reduced_tests.parquet  # abs path
taco_train_path=/lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered/codegen__deduped_taco_11.1k_l1e-5_h0.9_8.9k_reduced_tests.parquet  # abs path
# Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_500_sampled_200.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet
# Logic (train)
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_297_l1e-05_h0.9_sampled_117.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_653_l1e-05_h0.9_sampled_197.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_3.4k_l1e-5_h0.9_1.6k_sampled_1.6k.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_dataset_2.8k_l1e-5_h0.9_1.2k_sampled_1.2k.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_dataset_2.9k_l1e-5_h0.9_1.9k_sampled_1.9k.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_dataset_5.7k_l1e-5_h0.9_1.3k_sampled_1.3k.parquet
# Logic (test)
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_300_sampled_200.parquet
graph_test_path=${TEST_DATA_DIR}/logic__graph_logical_dataset_150_sampled_77.parquet
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_150_sampled_100.parquet
arcagi1_test_path=${TEST_DATA_DIR}/simulation__arcagi1_200.parquet
# Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_fixed_12.1k_processed_l1e-5_h0.9_3.8k_sampled_3.8k.parquet
# Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_500_sampled_200.parquet
# Table (train)
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_7.4k_l1e-5_h0.9_4.5k_sampled_4.5k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_2.9k_l1e-5_h0.9_1.6k_sampled_1.6k.parquet
# Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_300_sampled_200.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_300_sampled_200.parquet
# Stem (train)
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k_aggressively_filtered_sampled_3.6k.parquet
# Stem (test)
gpqa_diamond_test_path=${TEST_DATA_DIR}/stem__gpqa_198.parquet


train_files="['${math_train_path}',\
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

test_files="['${math_test_path}',\
'${aime_test_path}',\
'${amc_test_path}',\
'${humaneval_test_path}',\
'${mbpp_test_path}',\
'${livecodebench_test_path}',\
'${zebralogic_test_path}',\
'${graph_test_path}',\
'${ordering_puzzle_test_path}',\
'${arcagi1_test_path}',\
'${codeio_test_path}',\
'${multihier_test_path}',\
'${hitab_test_path}',\
'${gpqa_diamond_test_path}']"




# =================== Ray start ===================
# ray stop at all nodes
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ${CONDA_BIN_PATH}ray stop

sleep 10
# Remove existing Ray cluster
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    ${CONDA_BIN_PATH}ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --include-dashboard=True --block & #--temp-dir ${LOG_ROOT}/${SLURM_JOB_ID}/n0 &

sleep 10

# Start Ray worker nodes
for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS,STEM_LLM_JUDGE_URL=$STEM_LLM_JUDGE_URL \
        ${CONDA_BIN_PATH}ray start --address "$address_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block & #--temp-dir ${LOG_ROOT}/${SLURM_JOB_ID}/n${i} &    
done
sleep 10


# =================== Start RL training ===================
"${CONDA_BIN_PATH}python" -m verl.recipe.dapo.src.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=False \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation=right \
    data.max_prompt_length=$((1024*4)) \
    data.max_response_length=$((1024*8)) \
    data.train_batch_size=512 \
    data.gen_batch_size=512 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1024*30)) \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((1024*30)) \
    actor_rollout_ref.ref.log_prob_micro_batch_size=null \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((1024*30)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((1024*30)) \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.attention_dropout=0.0 \
    +actor_rollout_ref.model.override_config.embd_pdrop=0.0 \
    +actor_rollout_ref.model.override_config.resid_pdrop=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=async_dapo \
    reward_model.overlong_buffer.enable=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$worker_num \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=4 \
    +trainer.val_generations_to_log_to_wandb=30 \
    trainer.default_local_dir=/lustrefs/users/shibo.hao/checkpoints/${WANDB_PROJECT}/${WANDB_EXPERIMENT_NAME} \
    trainer.resume_mode=auto

# sleep 24 hours 
sleep 86400