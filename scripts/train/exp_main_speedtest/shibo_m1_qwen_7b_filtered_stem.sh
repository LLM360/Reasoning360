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
export STEM_LLM_JUDGE_URL=http://10.0.5.198:8000

head_node_ip=0.0.0.0
port=6379
address_head=$head_node_ip:$port

export worker_num=1
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
# export GLOO_SOCKET_IFNAME=ens10f0np0


# =================== Data Mixture (genie-25K)===================
WORKING_DIR=${HOME}/../zhuojun.cheng/Reasoning360
TRAIN_DATA_DIR=${WORKING_DIR}/data/train_guru15k
TEST_DATA_DIR=${WORKING_DIR}/data/test/test
# Math (train)
math_train_path1=${TRAIN_DATA_DIR}/math__merged_deduped_l1e-5_h0.9_60.0k_sampled_2.2k.parquet
math_train_path2=${TRAIN_DATA_DIR}/math__patch_merged_deduped_13.3k_l1e-5_h0.9_9.3k_sampled_334.parquet
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet
# Code (train)
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__deduped_leetcode2k_2.4k_l1e-5_h0.9_1.3k_sampled_177.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__deduped_livecodebench_599_l1e-5_h0.9_451_sampled_61.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__deduped_primeintellect_9.6k_l1e-5_h0.9_7.6k_sampled_1.0k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__deduped_taco_11.1k_l1e-5_h0.9_8.9k_sampled_1.2k.parquet
# Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_500_sampled_200.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet
# Logic (train)
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_297_l1e-05_h0.9_sampled_45.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_653_l1e-05_h0.9_sampled_77.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_3.4k_l1e-5_h0.9_1.6k_sampled_631.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_dataset_2.8k_l1e-5_h0.9_1.2k_sampled_485.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_dataset_2.9k_l1e-5_h0.9_1.9k_sampled_736.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_dataset_5.7k_l1e-5_h0.9_1.3k_sampled_523.parquet
# Logic (test)
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_300_sampled_200.parquet
graph_test_path=${TEST_DATA_DIR}/logic__graph_logical_dataset_150_sampled_77.parquet
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_150_sampled_100.parquet
arcagi1_test_path=${TEST_DATA_DIR}/simulation__arcagi1_200.parquet
# Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_fixed_12.1k_processed_l1e-5_h0.9_3.8k_sampled_2.5k.parquet
# Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_500_sampled_200.parquet
# Table (train)
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_7.4k_l1e-5_h0.9_4.5k_sampled_1.9k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_2.9k_l1e-5_h0.9_1.6k_sampled_645.parquet
# Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_300_sampled_200.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_300_sampled_200.parquet
# Stem (train)
webinstruct_train_path=/lustrefs/users/shibo.hao/Reasoning360-May/data/stem__web_3.6k_aggressively_filtered_sampled_2.5k.parquet
# Stem (test)
gpqa_diamond_test_path=${TEST_DATA_DIR}/stem__gpqa_198.parquet


train_files="['${webinstruct_train_path}']"

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


# =================== Model ===================
# LOCAL_MODEL_DIR=${HOME}/.cache/huggingface/hub
# BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
# BASE_MODEL=Qwen/Qwen2.5-3B
BASE_MODEL=${HOME}/Qwen2.5-7B-think
# BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

JOB_NAME="STEM"

# =================== Logging ===================
WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=DEBUG-${BASE_MODEL##*/}-${JOB_NAME}-AggressiveFilter-STEM
WANDB_API_KEY=633cdb1f1b9dfb2ae2681e47863635fe33b93a10

export WANDB_API_KEY=${WANDB_API_KEY}


# =================== Ray start ===================
# ray stop at all nodes
ray stop

echo "Ray stopped"

sleep 5
# Remove existing Ray cluster
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 64 --num-gpus 8 --include-dashboard=True --block &

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
enable_overlong_buffer=False
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
    trainer.nnodes=1 \
    trainer.nnodes=$worker_num \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    +trainer.val_generations_to_log_to_wandb=30 \
    trainer.resume_mode=auto