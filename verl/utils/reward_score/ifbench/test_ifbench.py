# test_ifbench.py
import sys
import os

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
sys.path.insert(0, project_root)

from .instructions_registry import INSTRUCTION_DICT
from . import compute_score

# Test data
example = {
    "data_source": "ood__ifbench",
    "prompt": [
        {"role": "user", "content": "If \"A woman in pink is sitting and enjoying the object she is holding.\" does that mean that \"Woman sitting.\"?\nOptions:\n- yes\n- it is not possible to tell\n- no Stream of thoughts: There should be 4 paragraphs. Paragraphs are separated with the markdown divider: *** In your response, the letter s should appear at least 20 times. Include keyword flight in your response."}
    ],
    "reward_model": {
        "style": "rule",
        "ground_truth": "[{'instruction_id': ['length_constraints:number_paragraphs', 'letters:letter_counting2', 'keywords:word_once'], 'kwargs': [{'num_paragraphs': 4}, {'letter': 's', 'let_frequency': 20, 'let_relation': 'at least'}, {'keyword': 'flight'}]}]"
    }
}

# Solution that meets the requirements
solution = """yes

The statement clearly indicates that a woman is sitting, which directly answers the question about whether a woman is sitting.

***

The woman in pink is specifically mentioned as sitting and enjoying an object, which provides sufficient information to conclude that a woman is indeed sitting in this scenario.

***

The keyword flight appears in this context as we discuss the woman's position and activities, demonstrating how various elements can be connected in meaningful ways.

***

The letter s appears multiple times throughout this response, satisfying the requirement for frequent usage while maintaining coherent and logical content structure."""

# Compute score
result = compute_score(
    solution_str=solution,
    ground_truth=example["reward_model"]["ground_truth"],
    extra_info=None
)

print(f"acc: {result['acc']}")
print(f"reward: {result['score']}")

# Verify requirements
print(f"\nVerification of requirements:")
print(f"Number of paragraphs: {solution.count('***') + 1}")
print(f"Number of occurrences of letter 's': {solution.count('s')}")
print(f"Contains 'flight': {'flight' in solution.lower()}")