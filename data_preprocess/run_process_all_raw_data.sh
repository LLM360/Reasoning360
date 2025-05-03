# Process all datasets

set -e

# Math datasets
python data_preprocess/math/dapo_or1_merge_dedup_apr30.py
python data_preprocess/math/bigmath_preview_filtered_mar21.py
python data_preprocess/math/deepscaler_preview.py

# Code datasets
python data_preprocess/codegen/leetcode2k.py
python data_preprocess/codegen/taco.py  # takes ~30 mins, uncomment to run
python data_preprocess/codegen/primeintellect.py  # takes ~1 hour, uncomment to run
python data_preprocess/codegen/humaneval.py
python data_preprocess/codegen/mbpp.py
python data_preprocess/codegen/livecodebench.py

# Logic datasets
## 1.zebra puzzle
python data_preprocess/logic/zebrapuzzle_gen/puzzle_generator.py --output_dir data/raw --num_puzzles 6000 --num_processes 64
cd ..
python data_preprocess/logic/process_zebrapuzzle_dataset.py
## 2.graph logic
pip install pybind11
pip install Faker==37.1.0
cd data_preprocess/logic/graph_dataset_gen/
python logic.py --num_samples 3000
cd ../../..  # return to Reasoning360
python data_preprocess/logic/process_graph_dataset.py
## ordering puzzle
python data_preprocess/logic/puzzle_gen.py --test True --num_puzzles 3000
python data_preprocess/logic/process_puzzles_dataset.py

# Simulation datasets
python data_preprocess/simulation/codeio.py --train-sample-size 10000 --test-sample-size 500
python data_preprocess/simulation/arcagi.py --name arcagi1
python data_preprocess/simulation/arcagi.py --name arcagi2

# Table datasets
python data_preprocess/table/multihier.py
python data_preprocess/table/hitab.py

# Stem datasets
python data_preprocess/stem_web/stem.py --parquet_out data/raw

echo "All data preprocessing complete!"