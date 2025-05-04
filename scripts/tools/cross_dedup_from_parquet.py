import datasets
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Callable, Any, Dict
import argparse
import os
import pandas as pd

from verl.utils.data_process.utils import save_dataset


def default_is_equal(text1: str, text2: str) -> bool:
    """Default equality check function."""
    return text1 in text2 or text2 in text1


def is_unique_item(item: Tuple[int, str, str], indexed_items: List[Tuple[int, str, str]]) -> bool:
    """Check if an item is unique compared to previous items."""
    idx, content, data_source = item
    for prev_idx, prev_content, prev_data_source in indexed_items:
        if prev_idx >= idx:
            break
        if default_is_equal(content, prev_content):
            print(f"========Found duplicate item========")
            print(f"[INFO] Found duplicate item: \n +[{data_source}] {content}\n +[{prev_data_source}] {prev_content}\n")
            return False
    return True


def process_item(item: Tuple[int, str, str], 
                all_indexed_items: List[Tuple[int, str, str]]) -> bool:
    """Process single item."""
    curr_idx, _, _ = item
    previous_items = [
        (i, content, data_source) for i, content, data_source in all_indexed_items 
        if i < curr_idx
    ]
    return is_unique_item(item, previous_items)


def deduplicate_datasets(
    input_data_names: List[str],
    data_dir: str = "data/train",
    is_equal_func: Callable[[str, str], bool] = default_is_equal,
    num_processes: int = mp.cpu_count() - 2
) -> List[datasets.Dataset]:
    """
    Deduplicate multiple datasets globally.
    
    Args:
        input_data_names: List of names of the input datasets
        output_data_name: Name of the output dataset
        is_equal_func: Function to determine if two items are equal
        num_processes: Number of processes for parallel processing
        
    Returns:
        List of deduplicated datasets
    """
    # Load all datasets and extract contents
    print("Loading datasets and extracting contents...")
    loaded_datasets = []
    indexed_contents = []
    global_idx = 0
    
    for data_name in input_data_names:
        data_path = os.path.join(data_dir, f"{data_name}.parquet")
        try:
            dataset = datasets.Dataset.from_parquet(data_path)
        except Exception as e:
            # for nested columns, e.g., livecodebench
            import polars as pl
            dataframe = pl.read_parquet(data_path).to_pandas()
            dataset = datasets.Dataset.from_pandas(dataframe)
            print(f"Loaded {data_name} from {data_path}, with {len(dataset)} items")
            
        
        if dataset[0]['extra_info'].get("original_question"):
            # math
            contents = [item['extra_info']['original_question'].lower().strip() for item in dataset]
        else:
            # code
            contents = [item['extra_info']['original_prompt'].lower().strip() for item in dataset]
        data_sources = [item['data_source'] for item in dataset]
        loaded_datasets.append((data_name, dataset))
        indexed_contents.extend([
            (global_idx + i, content, data_source) for i, (content, data_source) in enumerate(zip(contents, data_sources))
        ])
        global_idx += len(dataset)

    # Process in parallel
    print("Processing items...")
    with mp.Pool(num_processes) as pool:
        process_func = partial(process_item, all_indexed_items=indexed_contents)
        is_unique = list(tqdm(
            pool.imap(process_func, indexed_contents, chunksize=100),
            total=len(indexed_contents),
            desc="Deduplicating"
        ))

    # Add dedup column to each dataset
    print("Adding dedup column to datasets...")
    deduplicated_datasets = []
    start_idx = 0
    for dataset_name, dataset in loaded_datasets:
        end_idx = start_idx + len(dataset)
        dedup_mask = is_unique[start_idx:end_idx]
        deduplicated_datasets.append((dataset_name, dataset.add_column("is_unique", dedup_mask) ))
        start_idx = end_idx

    return deduplicated_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--input_data_names", type=str, nargs="+")
    parser.add_argument("--equal_func_name", type=str, default="contain", choices=["contain"])
    parser.add_argument("--domain", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--output_data_name", type=str, default="merged_deduped")
    parser.add_argument("--merge_data", type=bool, default=False)
    args = parser.parse_args()
    
    # Hardcoded input data for now
    # data_parquet_paths = args.data_parquet_paths
    input_data_names = [
        # f"math__bigmath_preview_filtered_mar21_157.3k",
        f"math__deepscaler_preview_40.3k",
        f"math__merged_deduped_dapo_or1_dataset_117.2k",
        # f"codegen__leetcode2k_2.4k",
        # f"codegen__taco_11.2k",
        # f"codegen__primeintellect_11.4k",
        # f"codegen__livecodebench_599",
    ]
    equal_func_maps = {
        "contain": lambda x, y: x in y or y in x
    }

    # Example with global deduplication
    deduplicated_datasets = deduplicate_datasets(
        input_data_names=input_data_names,
        data_dir=args.data_dir,
        is_equal_func=equal_func_maps[args.equal_func_name],
    )

    # Save the new dataset here
    total_items = 0
    total_unique_items = 0
    deduped_dataset = []
    for data_name, dataset in deduplicated_datasets:
        total_items += len(dataset)
        single_deduped_dataset = dataset.filter(lambda x: x['is_unique'])
        total_unique_items += len(single_deduped_dataset)
        deduped_dataset.append(single_deduped_dataset)
        print(len(single_deduped_dataset))
        if not args.merge_data:
            deduped_dataset_path = save_dataset(
                dataset=single_deduped_dataset,
                output_dir=args.data_dir,
                filename_prefix=f"{args.domain}__deduped_{data_name.split('__')[1]}",
                sample_size=None
            )
            print(f"Saved deduplicated dataset to {deduped_dataset_path}")

    if args.merge_data:
        # Merge into one dataset, but don't use concatenate_datasets
        merged_dataset = []
        for dataset in deduped_dataset:
            print(len(dataset))
            merged_dataset.extend(dataset)
        merged_dataset = datasets.Dataset.from_pandas(pd.DataFrame(merged_dataset))
        deduped_dataset_path = save_dataset(
            dataset=merged_dataset,
            output_dir=args.data_dir,
            filename_prefix=f"{args.domain}__{args.output_data_name}",
            sample_size=None
        )
        print(f"Saved deduplicated dataset to {deduped_dataset_path}")
    print(f"Total items: {total_items}")
    print(f"Total unique items: {total_unique_items}")
    print(f"Deduplication ratio: {total_unique_items / total_items}")
