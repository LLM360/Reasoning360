import reasoning_gym
import json
import re

def compute_score(solution_str, ground_truth, extra_info=None, item=None):
    """
    Compute the reward score for reasoning gym tasks.

    Args:
        solution_str (str): The model's response/solution string
        ground_truth (str or dict): The ground truth answer or entry dict
        extra_info (str or dict, optional): Should contain 'task' (as JSON string or dict)
        item (dict, optional): The full data item, for fallback

    Returns:
        dict: {"score": float, "acc": float}
    """
    task = None
    entry = None

    # 1. Parse extra_info
    extra_info_dict = {}
    metadata = None
    
    if extra_info:
        if isinstance(extra_info, str):
            try:
                extra_info_dict = json.loads(extra_info)
            except Exception:
                extra_info_dict = {}
        else:
            extra_info_dict = extra_info

        # Get task first
        task = extra_info_dict.get("task")
        entry = extra_info_dict.get("entry")

        # Handle metadata field if present
        if "metadata" in extra_info_dict:
            if isinstance(extra_info_dict["metadata"], str):
                try:
                    metadata = json.loads(extra_info_dict["metadata"])
                except Exception:
                    metadata = {}
            elif isinstance(extra_info_dict["metadata"], dict):
                metadata = extra_info_dict["metadata"]

    # 2. Try to get from item (fallback - this is rarely used in actual training)
    if not task and item and isinstance(item, dict):
        task = item.get("ability")

    # 3. Try to get from ground_truth
    if not task and isinstance(ground_truth, dict):
        task = ground_truth.get("task")
        entry = ground_truth

    if not task:
        raise ValueError("task must be provided in extra_info, item, or ground_truth dict.")

    # 4. Get scoring function
    scorer = reasoning_gym.get_score_answer_fn(task)

    # 5. Get entry
    if entry is None:
        entry = {"answer": ground_truth}

    # Build metadata field, prioritizing extra_info metadata
    if isinstance(entry, dict):
        if "metadata" not in entry or not isinstance(entry["metadata"], dict):
            entry["metadata"] = {}
        if metadata is not None:
            entry["metadata"].update(metadata)
        if task is not None:
            entry["metadata"]["task"] = task
        entry["metadata"]["solution_str"] = solution_str
        entry["metadata"]["ground_truth"] = ground_truth
        if extra_info is not None:
            entry["metadata"]["extra_info"] = extra_info
        if item is not None:
            entry["metadata"]["item"] = item

    # 6. Extract clean answer from solution_str
    clean_answer = extract_answer_from_solution(solution_str)
    
    # 7. Scoring with task-specific fixes
    debug_log_path = "reasoning_gym_debug.log"
    try:
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write("[DEBUG] solution_str: {}\n".format(solution_str))
            f.write("[DEBUG] clean_answer: {}\n".format(clean_answer))
            f.write("[DEBUG] ground_truth: {}\n".format(ground_truth))
            f.write("[DEBUG] task: {}\n".format(task))
            f.write("[DEBUG] metadata: {}\n".format(json.dumps(entry.get("metadata", {}), ensure_ascii=False, indent=2)))
            
            # Get raw score from reasoning_gym using clean answer
            raw_score = scorer(answer=clean_answer, entry=entry)
            
            # Apply task-specific corrections for known issues
            corrected_score = apply_task_specific_corrections(task, solution_str, ground_truth, raw_score)
            
            f.write("[DEBUG] raw_score: {}\n".format(raw_score))
            f.write("[DEBUG] corrected_score: {}\n".format(corrected_score))
            
        return {"score": float(corrected_score), "acc": float(corrected_score)}
    except Exception as e:
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(f"Error in reasoning gym scoring: {e}\n")
        return {"score": 0.0, "acc": 0.0}


def apply_task_specific_corrections(task, solution_str, ground_truth, raw_score):
    """
    Apply corrections for known issues in specific reasoning_gym tasks.
    
    Args:
        task (str): The task name
        solution_str (str): The model's solution
        ground_truth (str): The ground truth answer
        raw_score (float): The raw score from reasoning_gym
        
    Returns:
        float: The corrected score
    """
    
    # Fix for puzzle24: Convert partial credit (0.01) to 0.0 for wrong answers
    if task == "puzzle24":
        if raw_score == 0.01:
            # Only give 0.01 if the solution actually attempts the format but is wrong
            # Otherwise give 0.0 for completely invalid answers
            if is_valid_puzzle24_format(solution_str):
                return 0.01  # Keep partial credit for valid format but wrong calculation
            else:
                return 0.0   # No credit for invalid format
        return raw_score
    
    # Fix for game_of_life_halting: Implement proper scoring since reasoning_gym seems broken
    elif task == "game_of_life_halting":
        # The reasoning_gym library appears to have a bug for this task
        # Implement simple exact string matching as fallback
        if solution_str.strip().lower() == ground_truth.strip().lower():
            return 1.0
        else:
            return 0.0
    
    # For all other tasks, return the raw score
    return raw_score


def is_valid_puzzle24_format(solution_str):
    """
    Check if a solution string follows a valid puzzle24 format.
    Valid formats include mathematical expressions with +, -, *, /, (, ) and numbers.
    """
    import re
    
    # Remove whitespace
    solution = solution_str.strip()
    
    # Check if it contains only valid characters for mathematical expressions
    valid_chars = re.match(r'^[0-9+\-*/.() ]+$', solution)
    if not valid_chars:
        return False
    
    # Check if it contains at least some mathematical operators
    has_operators = any(op in solution for op in ['+', '-', '*', '/'])
    
    # Check if it contains numbers
    has_numbers = re.search(r'\d', solution)
    
    return bool(has_operators and has_numbers)


def extract_answer_from_solution(solution_str):
    """
    Extract the final answer from a solution string that may contain <think> and <answer> tags.
    
    Args:
        solution_str (str): The full solution string from the model
        
    Returns:
        str: The extracted answer, or the original string if no answer tags found
    """
    # Try to extract from <answer> tags first
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Return the last answer if multiple found
        return matches[-1].strip()
    
    # If no <answer> tags, try to extract everything after the last <think> block
    think_pattern = r'</think>\s*(.*?)$'
    think_matches = re.findall(think_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    
    if think_matches:
        # Clean up any remaining tags
        answer = think_matches[-1].strip()
        # Remove any remaining HTML-like tags
        answer = re.sub(r'<[^>]+>', '', answer).strip()
        if answer:
            return answer
    
    # If no structured format found, return the original string
    # This handles cases where the model generates direct answers without tags
    return solution_str.strip()