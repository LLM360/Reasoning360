# from verl.utils.reward_score import prime_math
from verl.utils.reward_score import math

def compute_score(model_output: str, ground_truth: str) -> bool:
    model_output = str(model_output).lower()
    ground_truth = str(ground_truth).lower()
    
    solution_str = model_output.split("</think>")[-1]
    answer_str = math.last_boxed_only_string(solution_str)
    if answer_str is not None:
        answer = math.remove_boxed(answer_str)
    else:
        answer = solution_str

    print(f">>> {answer}, {ground_truth}")
    if "|" not in ground_truth:
        # Single answer
        nanswer = answer.replace(",", "")
                        .replace("%", " / 100")
                        .replace("$", "")
                        .replace(":", "/")
        try:
            nanswer = float(eval(nanswer))
        except:
            nanswer = answer
        score = math.is_equiv(answer, ground_truth)
    else:
        # Multiple answers, in format "ans1|ans2|ans3"
        try:
            ground_truth = sorted([ans.strip() for ans in ground_truth.split("|")])
            answer = sorted([ans.strip() for ans in answer.split("|")])
            if len(ground_truth) != len(answer):
                score = 0
            else:
                score = 1
                for gt, res in zip(ground_truth, answer):
                    if not math.is_equiv(gt, res):
                        score = 0
                        break
        except Exception as e:
            print(f"Error postprocessing ground truth or response in tablereason.py: {e}")
            print(ground_truth)
            print(answer)
            score = 0

    return {"score": score, "acc": score}
