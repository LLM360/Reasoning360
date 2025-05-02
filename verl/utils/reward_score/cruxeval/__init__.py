from verl.utils.reward_score.cruxeval.utils import check_correctness

def compute_score(model_output: str, ground_truth: str) -> bool:
    model_output = str(model_output)
    # print(f">>> {model_output}")
    try:
        model_output = model_output.split("[ANSWER]")[1].strip()
        model_output = model_output.split("[/ANSWER]")[0].strip()
    except:
        model_output = model_output.strip("```python\n").strip("```")
    full_code = eval(ground_truth)["functional"] + "\n" + model_output
    # print(f">>> {full_code}")
    is_correct = 1 if check_correctness(full_code) else 0
    # print(f">>> {is_correct}")
    return {"score": is_correct, "acc": is_correct}