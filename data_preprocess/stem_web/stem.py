#!/usr/bin/env python3
import os
import json
import argparse
import random
import pprint

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np

# ------------------ prompt template ------------------ #
SYSTEM_MSG = (
    "You are a knowledgeable assistant. "
    "Answer the following questions and think step by step. Please output the final answer within \\boxed{}. "
)

def make_prompt(question: str):
    return [
        {"role": "user",   "content": SYSTEM_MSG + question.strip()}
    ]

# ------------------ filter predicate ------------------ #
def keep_example(example, max_answer_len):
    return (
        example.get("category") != "Mathematics"
        and example.get("difficulty") in ["University", "PhD"]
        and example.get("answer_type") not in ["Boolean", "Multiple Choice"]
        and len(example.get("answer", "").split()) > 0
    )

# --------------------------- main ------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_out",    default="samples_le30.json")
    p.add_argument("--parquet_out", default="webinstruct_le30.parquet")
    p.add_argument("--dataset",     default="TIGER-Lab/WebInstruct-verified")
    p.add_argument("--split",       default="train")
    p.add_argument("--tokenizer",   default="Qwen/Qwen3-8B")
    p.add_argument("--max_answer_len", type=int, default=30)
    args = p.parse_args()

    # 1) load and filter
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.filter(lambda ex: keep_example(ex, args.max_answer_len))
    print(f"Filtered down to {len(ds)} examples before length check")

    # 2) tokenize answers and select by token length
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    enc = tok(ds["answer"], padding=False, truncation=False)
    lengths = [len(ids) for ids in enc["input_ids"]]
    keep_idxs = [i for i, L in enumerate(lengths) if L <= args.max_answer_len]
    print(f"{len(keep_idxs)} examples with answer ≤ {args.max_answer_len} tokens")

    # 3) build JSON samples
    samples = []
    for i in keep_idxs:
        ex = dict(ds[i])
        ex["token_length"] = lengths[i]
        samples.append(ex)

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples)} samples to {args.json_out}")

    # 4) wrap into chat‐template records
    processed = []
    for ex in samples:
        processed.append({
            "data_source": "WebInstruct-le30",
            "prompt": make_prompt(ex["question"]),
            "raw_prompt": ex["question"],
            "ability": "QA",
            "apply_chat_template": True,
            "response": ex["answer"],
            "reward_model": {"ground_truth": ex["answer"]},
            "extra_info": {
                "category":     ex.get("category"),
                "difficulty":   ex.get("difficulty"),
                "answer_type":  ex.get("answer_type"),
                "token_length": ex["token_length"],
            },
        })

    # 5) show a few examples
    print("\n*** Example prompts (3 random rows) ***")
    for row in random.sample(processed, k=min(3, len(processed))):
        pprint.pprint({
            "prompt":      row["prompt"],
            "response":    row["response"],
            "extra_info":  row["extra_info"],
        }, compact=True, width=120)
        print("-" * 80)

    # 6) save to Parquet
    hf_ds = Dataset.from_list(processed)
    hf_ds.to_parquet(args.parquet_out)
    print(f"Saved {len(hf_ds)} rows to {args.parquet_out}")

    # 7) (optional) prompt‐length stats
    user_lens = []
    for row in processed:
        for msg in row["prompt"]:
            if msg["role"] == "user":
                user_lens.append(len(tok.encode(msg["content"], add_special_tokens=False)))
    arr = np.array(user_lens)
    print(f"User‐prompt token stats — "
          f"min:{arr.min()}, max:{arr.max()}, mean:{arr.mean():.1f}, std:{arr.std():.1f}")

if __name__ == "__main__":
    main()
