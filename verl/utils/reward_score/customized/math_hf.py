"""Reward functions for math.

Refer to https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
"""

import math
import re
from typing import Dict
from typing import Callable

from latex2sympy2_extended import NormalizationConfig
from math_verify import (
    LatexExtractionConfig,
    parse,
    verify,
)


def accuracy_reward(solution_str: str, ground_truth: str) -> float:
    """Reward function that checks if the completion is the same as the ground truth.

    Args:
        solution_str: The solution string to evaluate
        ground_truth: The ground truth string to compare against

    Returns:
        1.0 if the solution matches the ground truth, 0.0 otherwise
    """
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold_parsed) == 0:
        # If the gold solution is not parseable, we reward 1 to skip this example
        print("Failed to parse gold solution: ", ground_truth)
        return 1.0

    # We require the answer to be provided in correct latex (no malformed operators)
    answer_parsed = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )

    # Return 1 if the content is the same as the ground truth, 0 otherwise
    return float(verify(answer_parsed, gold_parsed))


def format_reward(solution_str: str, ground_truth: str) -> float:
    """Reward function that checks if the completion has a specific format.

    Args:
        solution_str: The solution string to evaluate
        ground_truth: The ground truth string (unused in this function)

    Returns:
        1.0 if the solution matches the format, 0.0 otherwise
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    match = re.match(pattern, solution_str, re.DOTALL | re.MULTILINE)
    return 1.0 if match else 0.0


def reasoning_steps_reward(solution_str: str, ground_truth: str) -> float:
    r"""Reward function that checks for clear step-by-step reasoning.

    Args:
        solution_str: The solution string to evaluate
        ground_truth: The ground truth string (unused in this function)

    Returns:
        Score between 0.0 and 1.0 based on number of reasoning steps

    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    count = len(re.findall(pattern, solution_str))
    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return min(1.0, count / 3)


def len_reward(solution_str: str, ground_truth: str) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Args:
        solution_str: The solution string to evaluate
        ground_truth: The ground truth string to compare against

    Returns:
        Reward between -0.5 and 0.5 based on length and correctness
    """
    # First check correctness
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

    if len(gold_parsed) == 0:
        print("Failed to parse gold solution: ", ground_truth)
        return 0.0  # Skip unparseable examples

    answer_parsed = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed=True,
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )

    is_correct = verify(answer_parsed, gold_parsed)

    # Use fixed reference lengths for single string case
    min_len = 50  # Example minimum reasonable length
    max_len = 500  # Example maximum reasonable length
    current_len = len(solution_str)

    if max_len == min_len:
        return 0.0

    lambda_val = 0.5 - (current_len - min_len) / (max_len - min_len)

    if is_correct:
        reward = lambda_val
    else:
        reward = min(0, lambda_val)

    return float(reward)


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
) -> Callable[[str, str], float]:
    def cosine_scaled_reward(solution_str: str, ground_truth: str) -> float:
        """Reward function that scales based on completion length using a cosine schedule.

        Args:
            solution_str: The solution string to evaluate
            ground_truth: The ground truth string to compare against

        Returns:
            Scaled reward based on correctness and length
        """
        gold_parsed = parse(
            ground_truth,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) == 0:
            print("Failed to parse gold solution: ", ground_truth)
            return 1.0  # Skip unparseable examples

        answer_parsed = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        is_correct = verify(answer_parsed, gold_parsed)
        gen_len = len(solution_str)

        # Apply cosine scaling based on length
        progress = gen_len / max_len
        cosine = math.cos(progress * math.pi)

        if is_correct:
            min_value = min_value_correct
            max_value = max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong

        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        return float(reward)

    return cosine_scaled_reward


def get_repetition_penalty_reward(
    ngram_size: int, max_penalty: float
) -> Callable[[str, str], float]:
    """Creates a reward function that penalizes repetitions.

    Args:
        ngram_size: size of the n-grams
        max_penalty: Maximum (negative) penalty for repetitions
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(solution_str: str, ground_truth: str) -> float:
        """Compute repetition penalty reward for a single solution.

        Args:
            solution_str: The solution string to evaluate
            ground_truth: The ground truth string (unused in this function)

        Returns:
            Negative reward based on repetition amount
        """
        if solution_str == "" or len(solution_str.split()) < ngram_size:
            return 0.0

        ngrams = set()
        total = 0
        for ng in zipngram(solution_str, ngram_size):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        return scaling * max_penalty

    return repetition_penalty_reward


REWARD_FUNCTION_MAP = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "len": len_reward,
}


class MathReward:
    def __init__(self, reward_fn_names: list[str]):
        for reward_fn_name in reward_fn_names:
            if reward_fn_name not in REWARD_FUNCTION_MAP:
                raise ValueError(f"Reward function {reward_fn_name} not found")
        self.reward_fn_names = reward_fn_names

    def compute_score(self, solution_str: str, ground_truth: str) -> float:
        total_reward = 0.0

        for reward_fn_name in self.reward_fn_names:
            reward_fn = REWARD_FUNCTION_MAP[reward_fn_name]
            reward = reward_fn(solution_str, ground_truth)
            total_reward += reward

        return total_reward
