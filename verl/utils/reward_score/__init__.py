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


def _default_compute_score(
    data_source, solution_str, ground_truth, extra_info=None, reward_metric=None
):
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)

        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    # NOTE: added by Reasoning360
    elif data_source in [
        "agentica-org/DeepScaleR-Preview-Dataset",
        "nanoverl/math",
        "nanoverl/aime",
        "nanoverl/amc",
        "nanoverl/minerva",
        "nanoverl/olympiad_bench",
        "orz_math_57k_collected.json",
        "examples/data_preprocess/orz_math_57k_collected.json",
        "SynthLabsAI/Big-Math-RL-Verified",
        "SDSB/deepscale_partial_mar21_filtered_basic",
        "SDSB/big_math_partial_mar21_filtered_basic",
        "SDSB/aime_repeated_8x",
        "SDSB/amc_repeated_4x",
    ]:
        if reward_metric == "prime_math" or reward_metric is None:
            from . import prime_math

            res = prime_math.compute_score(solution_str, ground_truth)[0]

        elif reward_metric == "math_llm_judge":
            from . import math_llm_judge
            res = math_llm_judge.compute_score(solution_str, ground_truth, extra_info=extra_info)
            
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
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['code']:
        from . import coder1
        res = coder1.compute_score(solution_str, ground_truth, extra_info=extra_info)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
