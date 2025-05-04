"""
Minimal STEM judge: every answer is graded by an external LLM.

Prerequisite:
  - Set env var STEM_LLM_JUDGE_URL to an OpenAI-compatible /v1/chat/completions.
  - Launch the service with your preferred model beforehand, e.g.

        vllm serve --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000
        export STEM_LLM_JUDGE_URL=http://127.0.0.1:8000/v1/chat/completions
"""

import os, re, requests
from typing import Tuple

# ------------ Prompt template ------------------------------------------------
JUDGE_TEMPLATE = """\
You are a strict grader for university-level STEM problems.

Question:
{QUESTION}

Reference Answer:
{REFERENCE_ANSWER}

Student Answer:
{STUDENT_ANSWER}

Carefully think and check whether the student answer is equivalent to the reference answer. 
You only need to refer to the reference answer to grade the student's answer. Sometimes the student's answer is expressed in a different way from the reference answer, but the meaning is the same, and you should still consider it correct. If they are not equivalent in mathematical sense, you should consider it incorrect.

<Final Grade>: CORRECT or INCORRECT

"""


# ------------ Core LLM call --------------------------------------------------
def _llm_judge(question: str, student: str, reference: str) -> bool:
    url_base = os.getenv("STEM_LLM_JUDGE_URL")
    if not url_base:
        raise EnvironmentError("STEM_LLM_JUDGE_URL not set")
    url = url_base.rstrip("/") + "/v1/chat/completions"

    prompt = JUDGE_TEMPLATE.format(
        QUESTION=question,
        STUDENT_ANSWER=student,
        REFERENCE_ANSWER=reference,
    )

    payload = {
        "model": "Qwen/Qwen3-30B-A3B",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]

    if False:
        print("=== LLM Judge RAW RESPONSE ===")
        import json as _json
        print(_json.dumps(data, indent=2, ensure_ascii=False))
        print("=== LLM Judge CONTENT ===")
        print(text)
        print("=== End of LLM Judge Reply ===\n")

    return "<Final Grade>: CORRECT" in text

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1:right_brace_idx].strip()


def match_answer(response):
    is_matched = False

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed
    
    return is_matched, response


# ------------ Public API -----------------------------------------------------
def compute_score(data_source: str,
                  model_output: str,
                  ground_truth: str,
                  extra_info: dict) -> Tuple[bool, float, str]:
    """
    Arguments
    ---------
    model_output : str   – agent's raw answer
    ground_truth : str   – reference answer
    extra_info   : dict  – MUST contain key "question"

    Returns
    -------
    (is_correct, score, normalized_student_answer)
        score is 1.0 if correct, else 0.0
    """
    model_output = str(model_output)
    ground_truth = str(ground_truth)
    is_matched, extracted_model_output = match_answer(model_output)
    question    = extra_info["question"]
    if is_matched==False:
        return 0.
    else:
        try:
            is_correct = _llm_judge(question, extracted_model_output, ground_truth)
            reward_log = ("answer: "+str(extracted_model_output)+"\n"+"gt: "+ground_truth+"\n"+"score: "+str(is_correct) +"\n"+"-"*50)
            # print(reward_log)
        except Exception as e:
            print(f"[judge-error] {e}")
            return 0.
    
# ---------------- Demo -------------------------------------------------------
if __name__ == "__main__":
    demo_item = {
        "question": "A cash-strapped young professional offers to buy your car "
                    "with four equal annual payments of $3,000, beginning 2 "
                    "years from today. Assuming you can invest at 10% and want "
                    "to receive $9,000 for the car, should you accept?",
        "answer": "$8,645.09"
    }
    agent_reply = "$8,645.09"           # pretend this comes from your agent
    score = compute_score(
        data_source="",
        model_output=agent_reply,
        ground_truth=demo_item["answer"],
        extra_info={"question": demo_item["question"]}
    )
    print("score =", score)
