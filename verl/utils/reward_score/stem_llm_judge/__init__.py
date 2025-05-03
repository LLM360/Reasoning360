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

<Final Grade>: CORRECT or INCORRECT

"""

# ------------ Helper: boxed{…} extraction -----------------------------------
_BOX_RE = re.compile(r"\\boxed\{([^}]+)\}")

def _extract(ans: str) -> str:
    """Return content inside \boxed{...} if present, else original string."""
    m = _BOX_RE.search(ans)
    return m.group(1).strip() if m else ans.strip()

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

    # 提取回复文本
    text = data["choices"][0]["message"]["content"]

    # 如果开启 DEBUG，就打印完整的 JSON 和最终文本
    if False:
        print("=== LLM Judge RAW RESPONSE ===")
        import json as _json
        print(_json.dumps(data, indent=2, ensure_ascii=False))
        print("=== LLM Judge CONTENT ===")
        print(text)
        print("=== End of LLM Judge Reply ===\n")

    return "<Final Grade>: CORRECT" in text
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

    student_ans = _extract(str(model_output))
    question    = extra_info["question"]

    try:
        is_correct = _llm_judge(question, student_ans, str(ground_truth))
    except Exception as e:
        # 如果 LLM 服务失败，保守给错并打印日志
        print(f"[judge-error] {e}")
        is_correct = False

    return float(is_correct)


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
