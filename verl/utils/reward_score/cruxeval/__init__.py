import re
from verl.utils.reward_score.cruxeval.utils import check_correctness


def compute_score(model_output: str, ground_truth: str) -> bool:
    model_output = str(model_output)
    # print(f">>> {model_output}")
    try:
        # remove content until </think>
        model_output = re.split(r"</think>", model_output)[1]
        # remove content between ```python and ```
        model_output = re.split(r"```python", model_output)[1]
        model_output = re.split(r"```", model_output)[0]
    except:
        model_output = model_output

    full_code = eval(ground_truth)["functional"] + "\n" + model_output
    # print(f">>> {full_code}")
    is_correct = 1 if check_correctness(full_code) else 0
    # print(f">>> {is_correct}")
    return {"score": is_correct, "acc": is_correct}
