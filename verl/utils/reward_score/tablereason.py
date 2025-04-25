from verl.utils.reward_score import prime_math
from verl.utils.reward_score import math

def compute_score(model_output: str, ground_truth: str) -> bool:
    model_output = str(model_output)
    ground_truth = str(ground_truth)
    _, response = prime_math.match_answer(model_output)
    score = math.is_equiv(response, ground_truth)
    return {"score": score, "acc": score}
