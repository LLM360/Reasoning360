#!/bin/bash
#SBATCH --job-name=server_math_llm_as_verifier
#SBATCH --partition=higherprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=slurm/serve_math_llm_as_verifier_%j.log
#SBATCH --error=slurm/serve_math_llm_as_verifier_%j.log


# (1) detect this node’s primary IP
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Detected NODE_IP = $NODE_IP"

# (2) export judge URL for downstream clients
export MATH_LLM_JUDGE_URL="http://${NODE_IP}:8000"
echo "MATH_LLM_JUDGE_URL=$MATH_LLM_JUDGE_URL"

# (3) launch the vLLM server bound to that IP
vllm serve openai/gpt-oss-120b --host "$NODE_IP" --data-parallel-size 8 --enable-expert-parallel