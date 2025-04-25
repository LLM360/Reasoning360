import re
import random
import ast
import operator
import json
import signal
import contextlib

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)



def validate_format(response):
    if not response:
        return False

    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return True
    return False


def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    # for llama chat template
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    # qwen chat template
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    

    if match:
        answer_text = match.group(1).strip()

        label_pattern = r'\b(entailment|contradiction|self_contradiction|neutral)\b'
        label_match = re.search(label_pattern, answer_text, re.IGNORECASE)

        if label_match:
            label = label_match.group(1).lower()
            return label

    return None

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1., timeout: float = 10.0):

    try:
        with time_limit(timeout):
            predicted_answer = extract_solution(solution_str)
            
            do_print = random.randint(1, 64) == 1

            if do_print:
                print("--------------------------------")
                print(f"Target: {ground_truth}")
                print(f"Extracted arrangement: {predicted_answer}")
                print(f"Solution string: {solution_str}")
                print(f"--------------------------------")

            if predicted_answer is None:
                if do_print:
                    print(f"No final arrangement found")
                return 0.0

            if predicted_answer == ground_truth:
                if do_print:
                    print(f"Correct answer")
                return score
            else:
                if do_print:
                    print(f"Incorrect answer")
                return format_score
    except TimeoutException:
        print("Computation timed out in folio_dataset")
        return 0.0
    except Exception as e:
        print(f"Error in compute_score in folio_dataset: {e}")
        return 0.0



