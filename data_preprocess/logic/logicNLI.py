import os
import datasets
import argparse
import re


InstructionFollow = "Given the above premise, determine the logical relationship between the hypothesis and the premise. Respond with 'self_contradiction' if the hypothesis directly contradicts information explicitly stated in the premise, 'contradiction' if it contradicts information that can be logically inferred from the premise, 'entailment' if it can be logically inferred from the premise, and 'neutral' if the hypothesis can neither be proved or disproved from the premise. Return the final answer in <answer> </answer> tags, for example <answer> entailment </answer>. "

def make_prefix(dp):
    premises = dp['premise']
    hypothesis = dp['hypothesis']
    label = dp['label']

    prefix = "Premise:\n {premises} \n Hypothesis:\n {hypothesis} \n"
    return prefix

def get_datasets(cache_dir : str):
    try:
        train_dataset = datasets.load_dataset("tasksource/LogicNLI", split="train")
        validation_dataset = datasets.load_dataset("tasksource/LogicNLI", split="validation")
        test_dataset = datasets.load_dataset("tasksource/LogicNLI", split="test")

        test_dataset = datasets.concatenate_datasets([validation_dataset, test_dataset])

        return train_dataset, test_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def make_map_fn(split, data_source):
    def process_fn(example, idx):
        question = make_prefix(example)
        answer = example['label']
        data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question + " " + InstructionFollow,
                }],
                "ability": "logical_reasoning",
                "apply_chat_template" : True,
                "reward_model": {
                        "style": "rule",
                        "ground_truth": answer
                    },
                "extra_info": {
                    'id': example['id'] if 'id' in example else str(idx),
                    'split': split,
                    'label': answer,
                }
            }
        if idx == 0:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data
    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='../../data/logicNLI_dataset', help='Local directory to save parquet files')
    parser.add_argument('--data_source', default='LogicNLI', help='Name of data source')
    parser.add_argument('--model_type', default='instruct', choices=['base', 'instruct'], help='Model type base or instruct')
    args = parser.parse_args()
    
    train_dataset, test_dataset = get_datasets(cache_dir=datasets.config.HF_DATASETS_CACHE)
    
    train_dataset = train_dataset.map(make_map_fn("train", args.data_source), with_indices=True)
    test_dataset = test_dataset.map(make_map_fn("test", args.data_source), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'logicNLI_dataset_train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'logicNLI_dataset_test.parquet'))

    print(f"Conversion complete. Files saved to {local_dir}")
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")






