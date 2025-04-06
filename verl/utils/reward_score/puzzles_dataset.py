import re
import random
import ast
import operator



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
    else:
        return None

    # Find the answer tag content
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    
    if matches:
        final_answer = matches[-1].group(1).strip()
        
        # Use regex to safely extract the bird names without eval
        bird_pattern = r'\[[\s\'\"]*([^\],]+)[\s\'\"]*(?:,[\s\'\"]*([^\],]+)[\s\'\"]*)*\]'
        bird_match = re.search(bird_pattern, final_answer)
        
        if bird_match:
            # Extract all bird names from the list
            bird_list_str = bird_match.group(0)
            # Extract individual birds, handling different formats
            birds = re.findall(r'[\'"]?([\w\s]+)[\'"]?', bird_list_str)
            # Clean up the extracted birds (removing list brackets, quotes, etc.)
            birds = [bird.lower().strip() for bird in birds if bird.strip() and bird.strip() not in ['[', ']']]
            return birds
        else:
            # If we can't extract using regex, try a safer approach
            try:
                # Add quotes around unquoted words to make it valid Python
                fixed_str = re.sub(r'(\[|\s*,\s*)(\w+)(\s*,|\])', r'\1"\2"\3', final_answer)
                return eval(fixed_str)
            except:
                # Last resort: just return as text
                return None
    else:
        return None


def compute_edit_distance(list1, list2):
    """
    Calculate edit distance between two lists.
    Returns the minimum number of operations (insertions, deletions, substitutions)
    required to transform list1 into list2.
    """
    # Create a matrix of size (len(list1)+1) x (len(list2)+1)
    dp = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]
    
    # Initialize the first row and column
    for i in range(len(list1) + 1):
        dp[i][0] = i
    for j in range(len(list2) + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                    dp[i][j-1],      # insertion
                                    dp[i-1][j-1])    # substitution
    
    return dp[len(list1)][len(list2)]

# granular reward function
def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for bird puzzles task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth.tolist()
    predicted_arrangement = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {ground_truth}")
        print(f"Extracted arrangement: {predicted_arrangement}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")
    if predicted_arrangement is None:
        if do_print:
            print(f"No final arrangement found")
        return 0
    
    
    # Validate format is correct
    if not validate_format(solution_str):
        if do_print:
            print(f"Format not followed properly")
        return format_score
        
    # Evaluate equation
    try:
        if isinstance(predicted_arrangement, list) and isinstance(target, list):            
            edit_distance = compute_edit_distance(predicted_arrangement, target)
            if do_print:
                print(f"Edit distance: {edit_distance}")
            max_possible_dist = max(len(predicted_arrangement), len(target))


        result = predicted_arrangement == target
        if result:
            if do_print:
                print(f"Correct arrangement")
            return score
        else:
            reward = max(format_score, score*(1.0 - (edit_distance / max_possible_dist)))
            if do_print:
                print(f"Wrong/Partially correct arrangement: predicted_arrangement = {predicted_arrangement}, target = {target}")
                print(f"Edit distance: {edit_distance}/{max_possible_dist}, Reward: {reward:.4f}")
            return reward
    except Exception as e:
        if do_print:
            print(f"Error evaluating result: {type(e).__name__}: {str(e)}")
            try:
                if isinstance(predicted_arrangement, list) and isinstance(target, list):
                    pred_norm = [str(p).lower().strip() for p in predicted_arrangement]
                    target_norm = [str(t).lower().strip() for t in target]
                    print(f"Normalized comparison: {pred_norm == target_norm}")
            except Exception as e2:
                print(f"Error during normalization attempt: {str(e2)}")
        return format_score 

def compute_accuracy(solution_str, ground_truth):
    target = ground_truth.tolist()
    predicted_arrangement = extract_solution(solution_str=solution_str)
    if predicted_arrangement is None:
        return 0
    if not validate_format(solution_str):
        return 0
    return int(predicted_arrangement == target)
