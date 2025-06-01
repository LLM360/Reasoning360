#!/bin/bash

# ============================ Crawl 'guru18k' run stats (part 1) ============================
# ===== part1 =====
guru18k_run_name_list=(
    "71131-zhoujun-math3k-guru18k-models--Qwen--Qwen2.5-7B-think"
    "72107-zhoujun-codegen3k-guru18k-models--Qwen--Qwen2.5-7B-think"
    "71130-zhoujun-logic3k-guru18k-models--Qwen--Qwen2.5-7B-think"
    "71136-zhoujun-table3k-guru18k-models--Qwen--Qwen2.5-7B-think"
    "71162-zhoujun-simulation3k-guru18k-models--Qwen--Qwen2.5-7B-think"
    "71138-zhoujun-stem3k-guru18k-models--Qwen--Qwen2.5-7B-think"
    "70459-zhoujun-mix-guru18k-models--Qwen--Qwen2.5-7B-think"
)
group_by=(
    "math3k"
    "codegen3k"
    "logic3k"
    "table3k"
    "simulation3k"
    "stem3k"
    "mix"
)
group_by_alias=(
    "guru18k_math3k"
    "guru18k_codegen3k"
    "guru18k_logic3k"
    "guru18k_table3k"
    "guru18k_simulation3k"
    "guru18k_stem3k"
    "guru18k_mix"
)
python crawl_wandb.py \
    --project "Reasoning360" \
    --entity "mbzuai-llm" \
    --wandb-api-key "7059672a7e5394620cedfffde2b14afbb7e3b58b" \
    --run-name-list "${guru18k_run_name_list[@]}" \
    --group-by ${group_by[@]} \
    --group-by-alias ${group_by_alias[@]} \
    --output-dir "wandb_data"
