from verl.utils.reward_score.cruxeval.utils import check_correctness

def compute_score(model_output: str, ground_truth: str) -> bool:
    model_output = str(model_output)
    model_output = model_output.split("[ANSWER]")[1].strip()
    model_output = model_output.split("[/ANSWER]")[0].strip()
    
    is_correct = check_correctness(model_output)
    return {"score": is_correct, "acc": is_correct}