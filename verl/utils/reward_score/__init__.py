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

from verl.utils.import_utils import deprecated


def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
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
    elif data_source.startswith('stem__gpqa'):
        from . import gpqa
        from . import supergpqa
        if "no_box" in data_source:
            res = gpqa.compute_score(solution_str, ground_truth)
        else:
            res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith('stem__supergpqa'):
        from . import supergpqa
        res = supergpqa.compute_score(solution_str, ground_truth)
    elif data_source.startswith('stem_web'):
        from . import stem_llm_judge
        res = stem_llm_judge.compute_score(data_source=data_source, model_output=solution_str, ground_truth=ground_truth, extra_info=extra_info)
    elif data_source in ["ood__ifeval"]:
        from . import ifeval
        res = ifeval.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in ["ood__livebench"]:
        from . import livebench
        res = livebench.compute_score(solution_str, ground_truth, extra_info=extra_info)
    # NOTE: above is added by Reasoning360
    elif data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
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
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle"]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)


__all__ = ["default_compute_score"]
