#!/bin/bash

# ============================ Crawl 7B 'guru_full' run stats ============================
yolo_7b_run_name_list=(
    "yolorun-azure-hpc-H200-instance-019.core42.ai-20250512_015607-Qwen2.5-7B-think-4node-guru-full-minibsz64"
)
group_by=(
    "yolorun-azure-hpc-H200-instance-019.core42.ai-20250512_015607-Qwen2.5-7B-think-4node-guru-full-minibsz64"
)
group_by_alias=(
    "qwen_7b_guru_full"
)
python crawl_wandb.py \
    --project "Reasoning360" \
    --entity "leoii22-uc-san-diego" \
    --wandb-api-key "633cdb1f1b9dfb2ae2681e47863635fe33b93a10" \
    --run-name-list ${yolo_7b_run_name_list[@]} \
    --group-by ${group_by[@]} \
    --group-by-alias ${group_by_alias[@]} \
    --output-dir "wandb_data"


# ============================ Crawl 32b 'guru_full' run stats ============================
yolo_32b_run_name_list=(
    "azure-hpc-H200-instance-010.core42.ai-20250512_013325-Qwen2.5-32B-think-16node-guru-full-minibsz64"
)
group_by=(
    "azure-hpc-H200-instance-010.core42.ai-20250512_013325-Qwen2.5-32B-think-16node-guru-full-minibsz64"
)
group_by_alias=(
    "qwen_32b_guru_full"
)
python crawl_wandb.py \
    --project "Reasoning360" \
    --entity "leoii22-uc-san-diego" \
    --wandb-api-key "633cdb1f1b9dfb2ae2681e47863635fe33b93a10" \
    --run-name-list "${yolo_32b_run_name_list[@]}" \
    --group-by ${group_by[@]} \
    --group-by-alias ${group_by_alias[@]} \
    --output-dir "wandb_data"


# ============================ Crawl 'guru15k' run stats (part 1) ============================
# ===== part1 =====
guru15k_run_name_list=(
    "22531-zhoujun-rl-guru15k-logic2.5k-models--Qwen--Qwen2.5-7B-think"
    "22518-zhoujun-rl-guru15k-math2.5k-models--Qwen--Qwen2.5-7B-think"
    "22946-zhoujun-rl-guru15k-table2.5k-models--Qwen--Qwen2.5-7B-think"
    "22530-zhoujun-rl-guru15k-simulation2.5k-models--Qwen--Qwen2.5-7B-think"
)
group_by=(
    "logic2.5k"
    "math2.5k"
    "table2.5k"
    "simulation2.5k"
)
group_by_alias=(
    "guru15k_logic2.5k"
    "guru15k_math2.5k"
    "guru15k_table2.5k"
    "guru15k_simulation2.5k"
)
python crawl_wandb.py \
    --project "Reasoning360" \
    --entity "mbzuai-llm" \
    --wandb-api-key "7059672a7e5394620cedfffde2b14afbb7e3b58b" \
    --run-name-list "${guru15k_run_name_list[@]}" \
    --group-by ${group_by[@]} \
    --group-by-alias ${group_by_alias[@]} \
    --output-dir "wandb_data"
# ===== part2 =====
guru15k_run_name_list=(
    "guru15k_mix_qwen_7b_shiboshibo"
    "guru15k_codegen_qwen_7b_shiboshibo"
)
group_by=(
    "mix"
    "codegen"
)
group_by_alias=(
    "guru15k_mix"
    "guru15k_codegen2.5k"
)
python crawl_wandb.py \
    --project "Reasoning360" \
    --entity "leoii22-uc-san-diego" \
    --wandb-api-key "633cdb1f1b9dfb2ae2681e47863635fe33b93a10" \
    --run-name-list "${guru15k_run_name_list[@]}" \
    --group-by ${group_by[@]} \
    --group-by-alias ${group_by_alias[@]} \
    --output-dir "wandb_data"

