import re
import random
import ast
import operator
import json



def validate_format(response):
    if not response:
        return False

    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return True
    return False

def extract_solution(solution_str):
    """Extract the final arrangement from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    # for llama chat template
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    # qwen chat template
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
        try:
            solution = ast.literal_eval(final_answer)
            return solution
        except (SyntaxError, ValueError):
            try:
                solution = json.loads(final_answer)
                return solution
            except json.JSONDecodeError:
                return None
        except Exception as e:
            print(f"Error extracting solution: {e}")
            return None
    else:
        return None


def compute_accuracy(answer, ground_truth):
    """
    compare grid level accuracy of the final answer w the ground truth
    """
    if not isinstance(answer, dict):
        return 0
    
    # num_objects
    num_rows = len(ground_truth["rows"])
    #num_attributes
    num_cols = len(ground_truth["header"])

    #total_correct_cells
    correct_cells = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if answer["rows"][i][j] == ground_truth["rows"][i][j]:
                correct_cells += 1
    #accuracy
    accuracy = correct_cells / (num_rows * num_cols)
    return accuracy

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):

    
    predicted_arrangement = extract_solution(solution_str)
    print(f"Type of predicted arrangement: {type(predicted_arrangement)}")
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Target: {ground_truth}")
        print(f"Extracted arrangement: {predicted_arrangement}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")

    if predicted_arrangement is None:
        if do_print:
            print(f"No final arrangement found")
        return 0.0


    if not validate_format(solution_str):
        if do_print:
            print(f"Format not followed properly")
        return 0.0

    try:
        accuracy = compute_accuracy(predicted_arrangement, ground_truth)
        if do_print:
            print(f"Accuracy: {accuracy}")
        return format_score + score * accuracy
    except Exception as e:
        if do_print:
            print(f"Error evaluating result: {type(e).__name__}: {str(e)}")
        return format_score








    
    


    
