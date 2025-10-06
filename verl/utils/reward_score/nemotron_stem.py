# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re


def extract_solution(solution_str, method='strict'):
    """
    Extract the final answer choice from an LLM's response to a multiple-choice nemotron_stem question.
    
    Args:
        solution_str (str): The full text response from the LLM
        method (str): 'strict' for exact format matching, 'flexible' for more lenient matching
        
    Returns:
        str: The extracted answer choice (A, B, C, or D) or None if not found
    """
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # First try to find answer in boxed format
        boxed_match = re.search(r"\\boxed\{([A-D])\}", solution_str)
        if boxed_match:
            return boxed_match.group(1)
        
        # Then try standard "Answer:" format
        answer_match = re.search(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", solution_str)
        if answer_match:
            return answer_match.group(1)
        
        # Try to find single letter answers at the end
        end_match = re.search(r"\b([A-D])\b(?!.*\b[A-D]\b)", solution_str)
        if end_match:
            return end_match.group(1)
            
        return None
        
    elif method == 'flexible':
        # Look for answers in parentheses
        answer = re.findall(r"\(([A-D])\)", solution_str)
        if answer:
            return answer[-1]  # Return the last found answer
        
        # Look for boxed answers
        boxed_answer = re.findall(r"\\boxed\{([A-D])\}", solution_str)
        if boxed_answer:
            return boxed_answer[-1]
        
        # Look for any A, B, C, D pattern
        general_answer = re.findall(r"\b([A-D])\b", solution_str)
        if general_answer:
            return general_answer[-1]
            
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1., extra_info=None):
    """The scoring function for nemotron_stem dataset.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth answer (A, B, C, or D)
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format when answer is extractable but wrong
        score: the score for the correct answer
        extra_info: additional information (not used in this implementation)
        
    Returns:
        dict: A dictionary containing 'score' and 'acc' keys
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return {'score': 0, 'acc': 0}
    else:
        if answer == ground_truth:
            return {'score': score, 'acc': 1.}
        else:
            return {'score': format_score, 'acc': 0.}