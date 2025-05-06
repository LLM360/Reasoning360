#!/bin/bash
set -x

# === Arguments ===
# Usage: bash run_generation_and_eval.sh <model_path>
datasets=("table__finqa_1.1k")
model_path=${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
n_samples=${2:1}

for dataset in ${datasets[@]}; do
    data_path=./data/test/${dataset}.parquet
    model_name=$(basename $model_path)
    save_path=./data/test/${dataset}_${model_name}_gen.parquet
    # === Generation ===
    echo "Generating with $model_name on ${dataset}..."
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=${data_path} \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path=${save_path} \
        model.path=${model_path} \
        rollout.temperature=0.6 \
        rollout.top_k=50 \
        rollout.top_p=0.95 \
        rollout.prompt_length=2048 \
        rollout.response_length=4096 \
        rollout.tensor_model_parallel_size=2 \
        rollout.gpu_memory_utilization=0.8

    # === Evaluation ===
    echo "Evaluating with $model_name on ${dataset}..."
    python3 -m verl.trainer.main_eval \
        data.path=${save_path} \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model
done