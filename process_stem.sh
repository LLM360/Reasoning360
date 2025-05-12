STEM_SIZE=2500
python postprocessing.py \
    --input_data_dir /lustrefs/users/shibo.hao/Reasoning360-May/data \
    --input_data_names stem__web_3.6k_aggressively_filtered \
    --output_data_dir /lustrefs/users/shibo.hao/Reasoning360-May/data \
    --target_sample_size $STEM_SIZE \
    --domain stem \
    --max_prompt_tokens 4096 > stem_sample.log