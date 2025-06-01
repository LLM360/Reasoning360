#!/bin/bash
#SBATCH --job-name=serve_gv
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=serve_gv_%j.log
#SBATCH --error=serve_gv_%j.err

# (1) load your environment
# module load python/3.9 cuda/11.7
# source ~/venv/bin/activate 7339 7340

# (2) detect this node’s primary IP
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Detected NODE_IP = $NODE_IP"

# (3) export judge URL for downstream clients
export STEM_LLM_JUDGE_URL="http://${NODE_IP}:8000"
echo "STEM_LLM_JUDGE_URL=$STEM_LLM_JUDGE_URL"

# (4) launch the vLLM server bound to that IP
/lustrefs/users/shibo.hao/miniforge3/envs/leo-reasoning360/bin/vllm serve TIGER-Lab/general-verifier --host "$NODE_IP" --data-parallel-size 8
