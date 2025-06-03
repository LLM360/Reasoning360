# Process all evaluation datasets

set -e

# Math datasets
python data_preprocess/math/dapo_or1_merge_dedup_apr30.py
python data_preprocess/math/bigmath_preview_filtered_mar21.py
python data_preprocess/math/deepscaler_preview.py

# Code datasets
python data_preprocess/codegen/humaneval.py
python data_preprocess/codegen/mbpp.py
python data_preprocess/codegen/livecodebench.py

# Stem datasets
# python data_preprocess/stem/gpqa.py
# python data_preprocess/stem/gpqa_diamond.py

# Logic datasets