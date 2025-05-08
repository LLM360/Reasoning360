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
    
    # math
    if data_source.startswith("math"):
        if reward_metric == "prime_math":
            from . import prime_math
            res = prime_math.compute_score(solution_str, ground_truth)
        elif reward_metric == "math_llm_judge":
            from . import math_llm_judge
            res = math_llm_judge.compute_score(
                solution_str, ground_truth, extra_info=extra_info
            )
        else:
            # Default
            from . import naive_dapo
            res = naive_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # code generation
    elif data_source.startswith('codegen'):
        from . import coder1
        res = coder1.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # simulation (code)
    elif data_source.startswith("simulation__codeio"):
        from . import codeio
        res = codeio.compute_score(solution_str, ground_truth)
    elif data_source.startswith("simulation__cruxeval"):
        from . import cruxeval
        res = cruxeval.compute_score(solution_str, ground_truth)
    # logic
    elif data_source.startswith("simulation__arcagi") or data_source.startswith("simulation__barc"):
        from . import arcagi
        res = arcagi.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__zebra_puzzle"):
        from . import zebra_puzzle
        res = zebra_puzzle.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__ordering_puzzle"):
        from . import puzzles_dataset
        res = puzzles_dataset.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__graph"):
        from . import graph_dataset
        res = graph_dataset.compute_score(solution_str, ground_truth)
    # table
    elif data_source.startswith("table"):
        # TODO: tmp placeholder using math_verify
        from . import tablereason
        res = tablereason.compute_score(solution_str, ground_truth)
    elif data_source in ["zebra_puzzle_dataset"]:
        from . import zebra_puzzle
        res = zebra_puzzle.compute_score(solution_str, ground_truth)
    elif data_source in ['ordering_puzzle_dataset']:
        from . import puzzles_dataset
        res = puzzles_dataset.compute_score(solution_str, ground_truth)
    elif data_source in ['graph_logical_dataset']:
        from . import graph_dataset
        res = graph_dataset.compute_score(solution_str, ground_truth)
    elif data_source in ['stem__gpqa', 'stem__gpqa_diamond']:
        # science
        from . import gpqa
        res = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ['stem_web'] :
        from . import stem_llm_judge
        res = stem_llm_judge.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
