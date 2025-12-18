import ast
import json

import numpy as np

from .instructions_registry import INSTRUCTION_DICT


def compute_score(solution_str, ground_truth, extra_info=None):
    """
    Compute the reward score for IFBench tasks based on ground truth constraints.

    Args:
        solution_str (str): Model's full output, may include a '<think>' section.
        ground_truth (str or list): Original ground_truth, either a Python-literal string or list of dicts.
        extra_info (dict, optional): Ignored for IFBench since constraints are in ground_truth.

    Returns:
        dict: {"score": float, "acc": bool}
    """
    # Strip off any thinking section
    if "</think>" in solution_str:
        answer = solution_str.split("</think>", 1)[1].strip()
    else:
        answer = solution_str.strip()

    # Parse ground_truth if it's a string
    if isinstance(ground_truth, str):
        try:
            gt_list = ast.literal_eval(ground_truth)
        except Exception:
            gt_list = json.loads(ground_truth)
    else:
        gt_list = ground_truth

    # Take the first set of constraints
    if not isinstance(gt_list, list) or not gt_list:
        return {"score": 0.0, "acc": False}
    first_item = gt_list[0]
    instruction_ids = first_item.get("instruction_id", [])
    kwargs_list = first_item.get("kwargs", [])

    # Evaluate each instruction
    results = []
    for instr_id, raw_args in zip(instruction_ids, kwargs_list):
        # Prepare args dict
        args = {} if raw_args is None else raw_args
        # Convert numpy and floats
        clean_args = {}
        for key, val in args.items():
            if isinstance(val, float):
                clean_args[key] = int(val)
            elif isinstance(val, np.ndarray):
                clean_args[key] = val.tolist()
            else:
                clean_args[key] = val

        # Build and check instruction
        instr_cls = INSTRUCTION_DICT[instr_id]
        instr = instr_cls(instr_id)
        instr.build_description(**clean_args)
        passed = bool(answer and instr.check_following(answer))
        results.append(passed)

    # Return 1.0 if all constraints are satisfied, 0.0 otherwise
    score = 1.0 if all(results) else 0.0
    return {"score": score, "acc": score == 1.0}
