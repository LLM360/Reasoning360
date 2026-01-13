import re
import random
import ast
import operator

from verl.utils.py_functional import timeout_limit


def extract_solution(solution_str):

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
        if re.search(r"^[A-Za-z]+$", final_answer):
            return final_answer
    else:
        return None


def compute_score(
    solution_str, ground_truth, extra_info: any = None, timeout: float = 10.0
):
    """The scoring function for graph dataset task.

    Args:
        solution_str: the solution text
        ground_truth: the correct answer
        timeout: maximum time in seconds to allow for computation
    """

    @timeout_limit(seconds=timeout)
    def _compute_with_timeout():
        if not isinstance(ground_truth, str):
            ground_truth_str = str(ground_truth)
        else:
            ground_truth_str = ground_truth

        target = ground_truth_str.lower()
        solution = extract_solution(solution_str)

        if solution:
            solution = solution.lower()
        else:
            return 0.0

        try:
            if target == solution:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            return 0.0

    try:
        score = _compute_with_timeout()
    except TimeoutError:
        print("Computation timed out in graph_dataset")
        score = 0.0
    except Exception as e:
        print(f"Error in compute_score in graph_dataset: {e}")
        score = 0.0

    return {"score": score, "acc": score}
