import os
import datasets
import polars as pl
import pandas as pd

from verl.utils.data_process.utils import save_dataset

data_name1 = "math__merged_deduped_258.4k"  # bigmath+dapo+or1+deepscaler
data_name2 = "math__merged_deduped_118.2k"  # dapo+or1+deepscaler

def load_dataset(data_name: str, data_dir: str) -> datasets.Dataset:
    data_path = os.path.join(data_dir, f"{data_name}.parquet")
    try:
        dataset = datasets.Dataset.from_parquet(data_path)
    except Exception as e:
    # for nested columns, e.g., livecodebench
        import polars as pl
        dataframe = pl.read_parquet(data_path).to_pandas()
        dataset = datasets.Dataset.from_pandas(dataframe)
    print(f"Loaded {data_name} from {data_path}, with {len(dataset)} items")
    return dataset
    
dataset1 = load_dataset(data_name1, "data/train")
dataset1 = dataset1.filter(lambda x: "dapo" in x["data_source"])
print(f"Filtered dataset1 to {len(dataset1)} items")
dataset2 = load_dataset(data_name2, "data/train")
print(f"Loaded dataset2 from {data_name2}, with {len(dataset2)} items")

# Create a set of original questions from dataset1
original_questions_dataset1 = set([example["extra_info"]["original_question"] for example in dataset1])

# Filter dataset2 to only include items not in dataset1
filtered_dataset = dataset2.filter(
    lambda example: example["extra_info"]["original_question"] not in original_questions_dataset1
)

print(f"Created filtered dataset with {len(filtered_dataset)} items")

# Save the filtered dataset
output_path = save_dataset(filtered_dataset, "data/train", f"math__patch_merged_deduped")
print(f"Saved filtered dataset to {output_path}")