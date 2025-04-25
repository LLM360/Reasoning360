# Guideline

The processed data will be saved in the `data/train` and `data/test` directories.
The naming convention is `<domain>__<dataset_name>_<dataset_size>`, where `<domain>` is one of `math`, `codegen`, `logic`, `simulation`, `table` for now.

# Usage
## Math
**BigMath**
```bash
python data_preprocess/math/bigmath_preview_filtered_mar21.py --train-sample-size <train_sample_size>
```
Note: currently, test sets such as aime24, amc, and math500 are also included within the same data processing file as BigMath. This means you must run the BigMath data pipeline, even if you do not intend to use the BigMath training set, in order to access those test sets.

**DeepScaler**
```bash
python data_preprocess/math/deepscaler_preview.py --train-sample-size <train_sample_size>
```


## Code
**leetcode2k**
```bash
python data_preprocess/codegen/leetcode2k.py --train-sample-size <train_sample_size>
```
**taco**
```bash
python data_preprocess/codegen/taco.py --train-sample-size <train_sample_size>
```
**primeintellect**
```bash
python data_preprocess/codegen/primeintellect.py --train-sample-size <train_sample_size>
```
**humaneval**
```bash
python data_preprocess/codegen/humaneval.py
```
**mbpp**
```bash
python data_preprocess/codegen/mbpp.py
```
**livecodebench**
```bash
python data_preprocess/codegen/livecodebench.py
```

## Logic
**zebra_puzzle_dataset**
```bash
python data_preprocess/logic/zebrapuzzle_gen/puzzle_generator.py --output_dir data/raw --num_puzzles <num_puzzles> --num_processes <num_processes>
cd ..
python data_preprocess/logic/process_zebrapuzzle_dataset.py
```

**graph_logical_dataset**
```bash
uv pip install pybind11
uv pip install Faker==37.1.0
cd data_preprocess/logic/graph_dataset_gen/
python logic.py --num_samples <num_samples>
cd ../../..  # return to Reasoning360
python data_preprocess/logic/process_graph_dataset.py
```

**ordering_puzzle_dataset**
```bash
uv pip install Faker==37.1.0
python data_preprocess/logic/puzzle_gen.py --test True --num_puzzles <num_puzzles>
python data_preprocess/logic/process_puzzles_dataset.py
```

**folio and logicNLI dataset**
```bash
python data_preprocess/logic/folio.py
python data_preprocess/logic/logicNLI.py
```

## Simulation
```bash
python data_preprocess/simulation/codeio.py --train-sample-size <train_sample_size> --test-sample-size <test_sample_size>
```

## Table
```bash
uv pip install gdown
python data_preprocess/table/multihier.py
```

# Add a new dataset
1. Add a new script in `data_preprocess/<domain>/<dataset_name>.py`
2. Add a new entry in `tests/data_process/test_data_preprocess.py`.
3. Run `pytest tests/data_process` to check the functionality of the data preprocessing scripts.
