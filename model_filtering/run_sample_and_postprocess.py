import argparse
import os
import datasets
import json
import transformers

from verl.utils.data_process.utils import save_dataset
from verl.utils.data_process.filter import LengthFilter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str, default="/mnt/weka/home/zhuojun.cheng/leo/Reasoning360/data/train_filtered")
    parser.add_argument("--input_data_names", type=str, nargs="+")
    parser.add_argument("--output_data_dir", type=str, default="/mnt/weka/home/zhuojun.cheng/leo/Reasoning360/data/train_guru15k")
    parser.add_argument("--target_sample_size", type=int, required=True)
    parser.add_argument("--domain", type=str, default="math", choices=["math", "codegen", "simulation", "logic", "table", "stem"])
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    args = parser.parse_args()

    # Load data
    print(args.input_data_names)
    total_sample_size = 0
    subset_sample_sizes = {}
    dataset_list = []
    for input_data_name in args.input_data_names:
        data_path = os.path.join(args.input_data_dir, f"{input_data_name}.parquet")
        print(f"Loading {input_data_name} from {data_path}")
        try:
            dataset = datasets.Dataset.from_parquet(data_path)
        except Exception as e:
            # for nested columns, e.g., livecodebench
            import polars as pl
            dataframe = pl.read_parquet(data_path).to_pandas()
            dataset = datasets.Dataset.from_pandas(dataframe)
            print(f"Loaded {input_data_name} from {data_path}, with {len(dataset)} items")
        subset_sample_sizes[input_data_name] = len(dataset)
        total_sample_size += len(dataset)
        dataset_list.append(dataset)
        
    # Sample to `target_sample_size` according to the ratio of each dataset
    os.makedirs(args.output_data_dir, exist_ok=True)
    for input_data_name, dataset in zip(args.input_data_names, dataset_list):
        # Filter by prompt length if specified
        if args.max_prompt_tokens is not None:
            tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
            prompt_length_filter = LengthFilter(tokenizer=tokenizer, max_length=args.max_prompt_tokens)
            print(f"Before filtering by prompt length {args.max_prompt_tokens}: num_items = {len(dataset)}")
            dataset = dataset.filter(lambda x: prompt_length_filter.check(x), num_proc=64)
            print(f"After filtering by prompt length {args.max_prompt_tokens}: num_items = {len(dataset)}")
            
        # Normalize the data source for 'logic' domain
        print("input_data_name: ", input_data_name)
        if "zebra_puzzle" in input_data_name or "ordering_puzzle" in input_data_name or "graph" in input_data_name:
            dataset = dataset.map(lambda item: {
                "data_source": "logic__" + item["data_source"]
            })
        
        # Refine the prompt for 'table' domain
        if "table" in input_data_name:
            original_instruction = "Please output the final answer within \\boxed{}. If there are multiple answers, please output them separated by |."
            refined_instruction = "Please output the final answer in \\boxed{}. If there are multiple answers, include them in a single box, separated by |."
            dataset = dataset.map(lambda item: {
                "prompt": [
                    {"role": "user", "content": item["prompt"][0]["content"].replace(original_instruction, refined_instruction)},
                ]
            })
        
        subset_sample_size = int(args.target_sample_size * subset_sample_sizes[input_data_name] / total_sample_size)

        sampled_dataset = dataset.shuffle(seed=42).select(range(subset_sample_size))
        print(f"{len(sampled_dataset)} items sampled from {input_data_name}")
        
        # Print the first 5 items
        for idx, item in enumerate(sampled_dataset):
            if idx < 5:
                print(f"========== {input_data_name} item#{idx} ==========")
                try:
                    print(json.dumps(item, indent=4))
                except:
                    print(item)
            else:
                break
        
        output_path = save_dataset(
            sampled_dataset, 
            args.output_data_dir, 
            f"{args.domain}__{input_data_name.split('__')[1]}_sampled", 
            subset_sample_size
            )
        print(f"Saving {output_path} with {subset_sample_size} items")

    print(f"Saved {args.output_data_dir} with {args.target_sample_size} items")
