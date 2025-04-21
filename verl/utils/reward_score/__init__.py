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


def _default_compute_score(data_source, solution_str, ground_truth, reward_metric=None, extra_info=None):
    if data_source.startswith("math"):
        if reward_metric == "prime_math" or reward_metric is None:
            from . import prime_math

            res = prime_math.compute_score(solution_str, ground_truth)[0]

        elif reward_metric == "math_llm_judge":
            from . import math_llm_judge

            res = math_llm_judge.compute_score(
                solution_str, ground_truth, extra_info=extra_info
            )

        elif reward_metric == "math_verify":
            from .orz.math_utils_sync import is_equal, solution2answer

            res = is_equal(
                solution2answer(solution_str),
                solution2answer(str(ground_truth)),
                math_mode="math_verify",
            )
        elif reward_metric == "boxed_math_verify":
            from .orz.math_utils_sync import is_equal, solution2answer

            def add_boxed(s):
                s = str(s)
                if "\\boxed" not in s:
                    s = f"\\boxed{{{s}}}"
                return s

            res = is_equal(
                add_boxed(solution2answer(solution_str)),
                add_boxed(solution2answer(str(ground_truth))),
                math_mode="math_verify",
            )
        elif reward_metric == "dapo":
            from . import naive_dapo
            res = naive_dapo.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith('codegen'):
        from . import coder1
        res = coder1.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("simulation"):
        from . import codeio
        res = codeio.compute_score(solution_str, ground_truth)
    elif data_source.startswith("table"):
        # TODO: tmp placeholder using math_verify
        from . import tablereason
        res = tablereason.compute_score(solution_str, ground_truth)
    elif data_source.startswith("logic__zebra"):
        from . import zebra_puzzle
        res = zebra_puzzle.compute_score(solution_str, ground_truth)
    elif data_source in ['ordering_puzzle_dataset']:
        from . import puzzles_dataset
        res = puzzles_dataset.compute_score(solution_str, ground_truth)
    elif data_source in ['graph_logical_dataset']:
        from . import graph_dataset
        res = graph_dataset.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
