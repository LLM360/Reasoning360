# Assume you are in the head node


# the first node is the head node
nodes=(
  "azure-hpc-H200-instance-010"
  "azure-hpc-H200-instance-029"
  "azure-hpc-H200-instance-030"
  "azure-hpc-H200-instance-031"
  "azure-hpc-H200-instance-034"
  "azure-hpc-H200-instance-092"
  "azure-hpc-H200-instance-094"
  "azure-hpc-H200-instance-095"
  "azure-hpc-H200-instance-096"
  "azure-hpc-H200-instance-097"
  "azure-hpc-H200-instance-098"
  "azure-hpc-H200-instance-100"
  "azure-hpc-H200-instance-101"
  "azure-hpc-H200-instance-102"
  "azure-hpc-H200-instance-103"
  "azure-hpc-H200-instance-104"
)

set -euo pipefail

######################## 1. Discover the node list ########################

worker_num=${#nodes[@]} || { echo "Need ≥1 node"; exit 1; }
head_node=${nodes[0]}
echo "[INFO] Using nodes: ${nodes[*]}"
echo "[INFO] Head node  : $head_node"

######################## 2. Cluster‑wide environment ########################
# **Everything here is copied verbatim from your sbatch header.**
export LD_LIBRARY_PATH=/usr/local/nccl-rdma-sharp-plugins/lib:$LD_LIBRARY_PATH
export UCX_TLS=dc
export UCX_NET_DEVICES=mlx5_ib0:1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=5
export NCCL_MIN_NCHANNELS=32
export NCCL_TOPO_FILE=/mnt/users/runner/scripts/ndv5-topo.xml
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_plm_rsh_no_tree_spawn=1
export OMPI_MCA_plm_rsh_num_concurrent=800
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_P2P_NET_CHUNKSIZE=$((512*1024))
export NCCL_PXN_DISABLE=1
export UCX_NET_DEVICES=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export STEM_LLM_JUDGE_URL=http://10.0.4.45:8000
export VLLM_USE_V1=0
export WANDB_API_KEY=633cdb1f1b9dfb2ae2681e47863635fe33b93a10
export CONDA_BIN_PATH=/lustrefs/users/shibo.hao/miniforge3/envs/Reasoning360-May/bin/
export HYDRA_FULL_ERROR=1

# Training specific variables

BASE_MODEL=${HOME}/Qwen2.5-32B-think
WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME="$(hostname)-$(date +%Y%m%d_%H%M%S)-${BASE_MODEL##*/}-16node-guru-full"

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

# train_files="['${barc_train_path}']"
# test_files="['${arcagi1_test_path}']"


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
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
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
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
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
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$worker_num \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=4 \
    +trainer.val_generations_to_log_to_wandb=30 \
    trainer.resume_mode=auto