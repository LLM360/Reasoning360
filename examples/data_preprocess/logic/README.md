zebra_puzzle_dataset

```bash
cd zebrapuzzle_gen
python puzzle_generator.py --num_puzzles 10000 --num_processes 32
python process_zebrapuzzle_dataset.py
```

graph_logical_dataset

```bash
cd ~/Reasoning360/examples/data_preprocess/logic
python logic.py --output_dir ../data/graph_dataset --output_file graph_search.json --num_samples 10000
python process_graph_dataset.py
```

ordering_puzzle_dataset

```bash
cd ~/Reasoning360/examples/data_preprocess/logic/graph_dataset_gen
uv pip install Faker==37.1.0

python puzzle_gen.py --num_puzzles 10000 --output_dir data/puzzles_dataset --output_file puzzles_dataset.json  --test True
python process_puzzles_dataset.py
```