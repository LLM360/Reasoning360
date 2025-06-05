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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get('use_shm', False))    
    # NOTE: added by Reasoning360. Use polars for livecodebench
    if 'livecodebench' in local_path:
        import polars as pl
        dataset = pl.read_parquet(local_path)
    else:
        dataset = pd.read_parquet(local_path)
    # NOTE: reasoning 360 prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    try:
        extra_info_data = dataset["extra_info"]
    except:
        extra_info_data = None

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    # NOTE: modified by Reasoning360. Use the default reward function if no customized one is provided.
    compute_score = get_custom_reward_fn(config) or default_compute_score

    # Create remote tasks
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in
    # NOTE: added by Reasoning360, count the pass rate
    passes = 0
    avg_pass = 0
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)
                # NOTE: added by Reasoning360, count the pass rate
                passes += score > 0
                avg_pass += score

    # NOTE: added by Reasoning360, print and count the pass rate.
    import os
    import json
    k = len(responses[-1])

    print(f"pass@{k}: {passes / total * 100.0}")
    print(f"pass@1_(avg{k}): {avg_pass / total * 100.0}")
    metric_output_path = config.data.path.replace(".parquet", "_metric.json")
    if os.path.exists(metric_output_path):
        with open(metric_output_path, "r") as f:
            metric_data = json.load(f)
        metric_data[f"pass@{k}"] = passes / total * 100.0
        metric_data[f"pass@1_(avg{k})"] = avg_pass / total * 100.0
    else:
        metric_data = {f"pass@{k}": passes / total * 100.0, f"pass@1_(avg{k})": avg_pass / total * 100.0}
    with open(metric_output_path, "w") as f:
        json.dump(metric_data, f, indent=4)
    ###

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    print(metric_dict)


if __name__ == "__main__":
    main()
