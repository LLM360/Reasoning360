#! /bin/bash

# math
MATH_SIZE=-1
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names math__merged_deduped_l1e-5_h0.9_60.0k math__patch_merged_deduped_13.3k_l1e-5_h0.9_9.3k \
    --output_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_guru_full \
    --target_sample_size $MATH_SIZE \
    --domain math > math_sample.log

# codegen (train)
CODEGEN_SIZE=-1
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names codegen__deduped_leetcode2k_2.4k_l1e-5_h0.9_1.3k codegen__deduped_livecodebench_599_l1e-5_h0.9_451 codegen__deduped_primeintellect_9.6k_l1e-5_h0.9_7.6k codegen__deduped_taco_11.1k_l1e-5_h0.9_8.9k \
    --output_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_guru_full \
    --target_sample_size $CODEGEN_SIZE \
    --domain codegen > codegen_sample.log

# logic (train)
LOGIC_SIZE=-1
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names logic__zebra_puzzle_dataset_5.7k_l1e-5_h0.9_1.3k simulation__arcagi1_297_l1e-05_h0.9 simulation__arcagi2_653_l1e-05_h0.9 simulation__barc_3.4k_l1e-5_h0.9_1.6k logic__ordering_puzzle_dataset_2.9k_l1e-5_h0.9_1.9k logic__graph_logical_dataset_2.8k_l1e-5_h0.9_1.2k \
    --output_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_guru_full \
    --target_sample_size $LOGIC_SIZE \
    --domain logic \
    --max_prompt_tokens 4096 > logic_sample.log

# simulation (train)
SIMULATION_SIZE=-1
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names simulation__codeio_fixed_12.1k_processed_l1e-5_h0.9_3.8k \
    --output_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_guru_full \
    --target_sample_size $SIMULATION_SIZE \
    --domain simulation \
    --max_prompt_tokens 4096 > simulation_sample.log

# table (train)
TABLE_SIZE=-1
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names table__hitab_7.4k_l1e-5_h0.9_4.5k table__multihier_2.9k_l1e-5_h0.9_1.6k \
    --output_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_guru_full \
    --target_sample_size $TABLE_SIZE \
    --domain table \
    --max_prompt_tokens 4096 > table_sample.log

# stem
STEM_SIZE=-1
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names stem__web_3.6k_aggressively_filtered \
    --output_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_guru_full \
    --target_sample_size $STEM_SIZE \
    --domain stem \
    --max_prompt_tokens 4096 > stem_sample.log
