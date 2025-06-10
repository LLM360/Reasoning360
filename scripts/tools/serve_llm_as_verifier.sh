#!/bin/bash
#SBATCH --job-name=server_llm_as_verifier
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=slurm/serve_llm_as_verifier_%j.log
#SBATCH --error=slurm/serve_llm_as_verifier_%j.err


# (1) detect this node’s primary IP
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Detected NODE_IP = $NODE_IP"

# (2) export judge URL for downstream clients
export STEM_LLM_JUDGE_URL="http://${NODE_IP}:8000"
echo "STEM_LLM_JUDGE_URL=$STEM_LLM_JUDGE_URL"

# (3) launch the vLLM server bound to that IP
vllm serve TIGER-Lab/general-verifier --host "$NODE_IP" --data-parallel-size 8
