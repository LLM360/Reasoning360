# Process all datasets

# Math datasets
python data_preprocess/math/bigmath_preview_filtered_mar21.py --train-sample-size 10000
python data_preprocess/math/deepscaler_preview.py --train-sample-size 10000

# Code datasets
python data_preprocess/codegen/leetcode2k.py
python data_preprocess/codegen/taco.py
# python data_preprocess/codegen/primeintellect.py --train-sample-size 5000
python data_preprocess/codegen/humaneval.py
python data_preprocess/codegen/mbpp.py
python data_preprocess/codegen/livecodebench.py

# Logic datasets
cd data_preprocess/logic/zebra_puzzle_gen
python puzzle_generator.py --num_puzzles 2500 --num_processes 64
cd ..
python process_zebrapuzzle_dataset.py --train_size 0.88 --test_size 0.12

# Simulation datasets
python data_preprocess/simulation/codeio.py --train-sample-size 2500 --test-sample-size 500

# Table datasets
python data_preprocess/table/multihier.py

echo "All data preprocessing complete!"