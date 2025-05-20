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

import os
import json
import math
import hydra
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import (
    math,
    gsm8k,
    cruxeval,
    tablereason,
    naive_dapo,
    coder1,
    gpqa,
    supergpqa,
    arcagi,
    ifeval,
    livebench,
    zebra_puzzle,
    graph_dataset,
    codeio,
)
import pandas as pd
import numpy as np
from tqdm import tqdm


def select_reward_fn(data_source):
    if data_source == "lighteval/MATH":
        return math.compute_score
    elif data_source.startswith("simulation__cruxeval"):
        return cruxeval.compute_score
    elif data_source.startswith("table"):
        return tablereason.compute_score
    # math
    elif data_source in ["math__aime_repeated_8x", "math__math", "math__olympiad_bench", "math__aime2025_repeated_8x"]:
        return naive_dapo.compute_score
    # code gen
    elif data_source in [
        "codegen__humaneval",
        "codegen__mbpp",
        "codegen__livecodebench",
    ]:
        return coder1.compute_score
    elif data_source in ['stem__gpqa', 'stem__gpqa_diamond']:
        return supergpqa.compute_score
    elif data_source in ['stem__gpqa_diamond_no_box']:
        return gpqa.compute_score
    elif data_source == "stem__supergpqa":
        return supergpqa.compute_score
    elif data_source in ["simulation__arcagi1", "simulation__barc"]:
        return arcagi.compute_score
    elif data_source in ["ood__ifeval"]:
        return ifeval.compute_score
    elif data_source in ["ood__livebench"]:
        return livebench.compute_score
    elif data_source.startswith("logic__zebra_puzzle"):
        return zebra_puzzle.compute_score
    elif data_source.startswith("logic__graph_logical_dataset"):
        return graph_dataset.compute_score
    elif data_source.startswith("simulation__codeio"):
        return codeio.compute_score
    else:
        raise NotImplementedError(f"Data source {data_source} not implemented")


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    if 'livecodebench' in local_path:
        import polars as pl
        dataset = pl.read_parquet(local_path)
    else:
        dataset = pd.read_parquet(local_path)
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    try:
        extra_info_data = dataset["extra_info"]
    except:
        extra_info_data = None

    passes = 0
    avg_pass = 0

    total = len(dataset)

    for i in tqdm(range(total), desc="evaluating..."):
        response_lst = responses[i]
        data_source = data_sources[i]
        # select reward score based on data_source
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data["ground_truth"]
        extra_info = extra_info_data[i] if extra_info_data is not None else None
        score_lst = []
        for r in response_lst:
            score = reward_fn(r, ground_truth, extra_info=extra_info)
            score_lst.append(score['acc'])

        max_score = np.max(score_lst)
        avg_pass += np.mean(score_lst)

        if max_score > 0:
            passes += 1

    print(f"pass@1: {passes / total * 100.0}")
    print(f"avg_pass: {avg_pass / total * 100.0}")
    
    metric_output_path = config.data.path.replace(".parquet", "_metric.json")
    if os.path.exists(metric_output_path):
        with open(metric_output_path, "r") as f:
            metric_data = json.load(f)
        metric_data["pass@1"] = passes / total * 100.0
        metric_data["avg_pass"] = avg_pass / total * 100.0
    else:
        metric_data = {"pass@1": passes / total * 100.0, "avg_pass": avg_pass / total * 100.0}
    with open(metric_output_path, "w") as f:
        json.dump(metric_data, f, indent=4)

if __name__ == "__main__":
    main()
