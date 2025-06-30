# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline evaluation script for generated sequences using a reward model and ground truth verifier.

This script reads a parquet file containing generated sequences and (optionally) ground truth,
computes reward scores for each response, and calculates pass@k metrics using an unbiased estimator.
Results are saved as a JSON file for further analysis.

Usage:
    python main_eval.py
"""

import json
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


# --------------------------------------------------------------------------- #
# Unbiased pass@k estimator
#    Formula: 1 - C(n-c, k) / C(n, k)
# --------------------------------------------------------------------------- #
def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute the unbiased pass@k estimate as described in Chen et al. (2021).

    Args:
        n (int): Total number of generated samples for this problem.
        c (int): Number of correct samples (score == 1.0).
        k (int): Target k value.

    Returns:
        float: Unbiased pass@k estimate.

    Raises:
        ValueError: If k > n.
    """
    if k > n:
        raise ValueError(f"k = {k} cannot be greater than n = {n}")
    if n - c < k:  # Not enough incorrect samples for k => pass@k = 1
        return 1.0
    prod = 1.0
    # ∏_{j=n-c+1}^{n} (1 - k / j)  ==  C(n-c, k) / C(n, k)
    for j in range(n - c + 1, n + 1):
        prod *= 1.0 - k / j
    return 1.0 - prod


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data, extra_info):
    """
    Ray remote function to process a single data item.

    Args:
        reward_fn (callable): Reward function to evaluate responses.
        data_source: The data source for this item.
        response_lst (list): List of generated responses.
        reward_data (dict): Reward model data, including ground truth.
        extra_info: Any extra information for scoring.

    Returns:
        tuple: (data_source, score_lst) where score_lst is a list of scores for each response.
    """
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth, extra_info) for r in response_lst]
    score_lst = [s["score"] for s in score_lst]
    return (
        data_source,
        score_lst,  # a list of scores for each response
    )


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    """
    Main evaluation entry point. Loads data, computes reward scores, and calculates pass@k metrics.

    Args:
        config: Hydra configuration object.
    """
    # Copy data to local (optionally using shared memory)
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))

    # Load dataset using polars for livecodebench, otherwise pandas
    if "livecodebench" in local_path:
        import polars as pl

        dataset = pl.read_parquet(local_path)
    else:
        dataset = pd.read_parquet(local_path)

    # Extract relevant columns from the dataset
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    try:
        extra_info_data = dataset["extra_info"]
    except Exception:
        extra_info_data = None

    total = len(dataset)

    # Initialize Ray for distributed processing
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # Prepare to collect per-data-source rewards
    data_source_reward = defaultdict(list)
    # Use custom reward function if provided, otherwise default
    compute_score = get_custom_reward_fn(config) or default_compute_score

    # Create Ray remote tasks for each data item
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i], extra_info_data[i] if extra_info_data is not None else dict()) for i in range(total)]

    # Compute max_k (number of responses per item) and candidate k values (powers of 2)
    if isinstance(responses, pd.Series) or isinstance(responses, pl.Series):
        max_k = len(responses.to_list()[-1])
    else:
        # numpy array
        max_k = len(responses.tolist()[-1])
    candidate_ks = [2**i for i in range(int(np.log2(max_k)) + 1) if 2**i <= max_k]
    pass_k_stat = {k: 0 for k in candidate_ks if k <= max_k}
    avg_pass = 0  # Sum of average scores for all items

    # Process results as they become available
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Wait for Ray tasks to complete
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score_lst = ray.get(result_id)
                # Count the number of correct responses (score == 1.0)
                pass_count = sum(1 for score in score_lst if score == 1)
                avg_score = float(np.mean(score_lst))
                avg_pass += avg_score
                data_source_reward[data_source].append(avg_score)
                pbar.update(1)

                # For each candidate k, update unbiased pass@k statistics
                for k_val, _ in enumerate(score_lst, start=1):
                    if k_val in candidate_ks:
                        pass_k_stat[k_val] += unbiased_pass_at_k(max_k, pass_count, k_val)

    # Prepare output metrics
    metric_output_path = config.data.path.replace(".parquet", "_metric.json")
    metric_data = {
        # Unbiased pass@k for each candidate k
        **{f"pass@{k_val}": pass_k_stat[k_val] / total * 100.0 for k_val in candidate_ks},
        # Traditional average pass@1 metric
        f"pass@1_(avg{max_k})": avg_pass / total * 100.0,
    }
    # Save metrics to JSON file
    with open(metric_output_path, "w") as f:
        json.dump(metric_data, f, indent=4)

    print(metric_data)

    # Print per-data-source average scores
    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score(avg@k)/{data_source}"] = float(np.mean(rewards))

    print(metric_dict)


if __name__ == "__main__":
    main()
