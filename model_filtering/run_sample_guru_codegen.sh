CODEGEN_SIZE=-1
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names codegen__deduped_leetcode2k_2.4k_l1e-5_h0.9_1.3k \
    --output_data_dir /lustrefs/users/shibo.hao/Reasoning360-May/data/ \
    --target_sample_size $CODEGEN_SIZE \
    --domain codegen > codegen_leetcode_sample.log
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names codegen__deduped_livecodebench_599_l1e-5_h0.9_451 \
    --output_data_dir /lustrefs/users/shibo.hao/Reasoning360-May/data/ \
    --target_sample_size $CODEGEN_SIZE \
    --domain codegen > codegen_livecodebench_sample.log
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names codegen__deduped_primeintellect_9.6k_l1e-5_h0.9_7.6k \
    --output_data_dir /lustrefs/users/shibo.hao/Reasoning360-May/data/ \
    --target_sample_size $CODEGEN_SIZE \
    --domain codegen > codegen_primeintellect_sample.log
python model_filtering/run_sample_and_postprocess.py \
    --input_data_dir /lustrefs/users/zhuojun.cheng/Reasoning360/data/train_filtered \
    --input_data_names codegen__deduped_taco_11.1k_l1e-5_h0.9_8.9k \
    --output_data_dir /lustrefs/users/shibo.hao/Reasoning360-May/data/ \
    --target_sample_size $CODEGEN_SIZE \
    --domain codegen > codegen_taco_sample.log