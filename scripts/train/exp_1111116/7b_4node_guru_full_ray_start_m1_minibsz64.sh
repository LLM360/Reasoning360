# azure-hpc-H200-instance-[010,029-031,034,092,094-095]

#!/bin/bash

# ========= USER CONFIG =========
nodes=(
  "azure-hpc-H200-instance-105"
  "azure-hpc-H200-instance-106"
  "azure-hpc-H200-instance-107"
  "azure-hpc-H200-instance-161"
)

# nodes=("azure-hpc-H200-instance-010" "azure-hpc-H200-instance-029")
#!/usr/bin/env bash
# launch_ray_rl.sh
#
# Starts a 4‑node Ray cluster and kicks off the VERL‑Dapo RL run
# mirrored from the original sbatch script.
#
# Usage (interactive allocation, password‑less SSH):
#   ./launch_ray_rl.sh node1 node2 node3 node4
#   # or, if SLURM exported $SLURM_NODELIST:
#   ./launch_ray_rl.sh
#
# ────────────────────────────────
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


remote() {
  host=$1; shift
  # allow any characters after the prefix, then the “=”
  exports=$(env | grep -E '^(LD_LIBRARY_PATH|UCX_[^=]*|CUDA_DEVICE_[^=]*|NCCL_[^=]*|GLOO_[^=]*|OMPI_[^=]*|CUDA_DEVICE_MAX_CONNECTIONS|STEM_LLM_JUDGE_URL|VLLM_USE_V1|WANDB_[^=]*)=' \
            | sed 's/^/export /')
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$host" "bash -lc '$exports; $*'"
}


######################## 5. Ray cleanup ########################
echo "[INFO] Stopping any previous Ray cluster ..."
for h in "${nodes[@]}"; do
    remote "$h" "${CONDA_BIN_PATH}ray stop --force || true"
    remote "$h" "rm -rf /tmp/ray/ray_current_cluster"
done

######################## 6. Ray bootstrap ########################
port=6379
head_ip=$(ssh "$head_node" hostname --ip-address | awk '{print $1}')
address="$head_ip:$port"
echo "[INFO] Head node IP: $head_ip"

echo "[INFO] Starting Ray HEAD on $head_node"
remote "$head_node" "nohup ${CONDA_BIN_PATH}ray start --head --node-ip-address=$head_ip --port=$port \
                     --num-cpus 96 --num-gpus 8 --include-dashboard=True --block \
                     > ray_head.log 2>&1 &"

sleep 10
for ((i=1;i<worker_num;i++)); do
    w=${nodes[$i]}
    echo "[INFO] Starting Ray WORKER on $w"
    remote "$w" "nohup ${CONDA_BIN_PATH}ray start --address $address --num-cpus 96 --num-gpus 8 --block \
                 > ray_worker.log 2>&1 &"
done
sleep 20
echo "[INFO] Ray cluster ready -> $address"
