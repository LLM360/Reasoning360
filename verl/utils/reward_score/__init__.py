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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    
    reward_metric = extra_info.get("reward_metric", None)
    
    if data_source.startswith("math"):
        
        if reward_metric == "prime_math":
            from . import prime_math
            res = prime_math.compute_score(solution_str, ground_truth)[0]

        elif reward_metric == "math_llm_judge":
            from . import math_llm_judge
            res = math_llm_judge.compute_score(
                solution_str, ground_truth, extra_info=extra_info
            )

        else:
            from . import naive_dapo
            res = naive_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info)
    
    elif data_source.startswith('codegen'):
        from . import coder1
        res = coder1.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("simulation__arcagi") or data_source.startswith("simulation__barc"):
        from . import arcagi
        res = arcagi.compute_score(solution_str, ground_truth)
    elif data_source.startswith("simulation__codeio"):
        from . import codeio
        res = codeio.compute_score(solution_str, ground_truth)
    elif data_source.startswith("table"):
        # TODO: tmp placeholder using math_verify
        from . import tablereason
        res = tablereason.compute_score(solution_str, ground_truth)
    elif data_source in ["logic__zebra_puzzle_dataset"]:
        from . import zebra_puzzle
        res = zebra_puzzle.compute_score(solution_str, ground_truth)
    elif data_source in ['logic__ordering_puzzle_dataset']:
        from . import puzzles_dataset
        res = puzzles_dataset.compute_score(solution_str, ground_truth)
    elif data_source in ['logic__graph_logical_dataset']:
        from . import graph_dataset
        res = graph_dataset.compute_score(solution_str, ground_truth)
    elif data_source in ['stem__gpqa']:
        from . import gpqa
        res = gpqa.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        print(f"[DEBUG] res is a dict, from data source: {data_source}")
        return res
    elif isinstance(res, (int, float, bool)):
        print(f"[DEBUG] res is a int, float, or bool, from data source: {data_source}")
        return float(res)
    else:
        print(f"[DEBUG] res is a list, from data source: {data_source}")
        return float(res[0])
