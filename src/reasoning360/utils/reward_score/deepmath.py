# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Optional, Union


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> float:
    """Compute the score for DeepMath dataset solutions.
    
    Args:
        solution_str: The model's solution/answer
        ground_truth: The correct answer from the dataset
        extra_info: Optional additional information (e.g., difficulty, topic)
    
    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    try:
        # Extract answer from solution if it's in boxed format
        extracted_answer = extract_boxed_answer(solution_str)
        if extracted_answer is None:
            # Try to extract from common answer patterns
            extracted_answer = extract_answer_patterns(solution_str)
        
        if extracted_answer is None:
            # Use the full solution string as last resort
            extracted_answer = solution_str.strip()
        
        # Normalize both answers for comparison
        normalized_solution = normalize_math_answer(extracted_answer)
        normalized_ground_truth = normalize_math_answer(ground_truth)
        
        # Check if answers are equivalent
        if is_equivalent(normalized_solution, normalized_ground_truth):
            return 1.0
        
        # Additional check for numerical equivalence
        if is_numerically_equivalent(normalized_solution, normalized_ground_truth):
            return 1.0
            
        return 0.0
    except Exception as e:
        print(f"Error in DeepMath scoring: {e}")
        return 0.0


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    # Look for the last boxed expression
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    
    # Also check for \boxed without braces
    pattern2 = r"\\boxed\s+([^\s]+)"
    matches2 = re.findall(pattern2, text)
    if matches2:
        return matches2[-1]
    
    return None


def extract_answer_patterns(text: str) -> Optional[str]:
    """Extract answer from common answer patterns."""
    patterns = [
        r"(?:final answer|answer)[\s:]*(?:is)?[\s:]*([^\n.]+)",
        r"(?:evaluates to|equals to|is equal to)[\s:]*([^\n.]+)",
        r"therefore[\s,]+([^\n.]+)",
        r"thus[\s,]+([^\n.]+)",
        r"hence[\s,]+([^\n.]+)",
        r"=\s*([^\n]+)$",  # Last equals sign
        r"(?:limit|integral|sum|product)[\s\w]*(?:evaluates to|is|equals)[\s:]*([^\n.]+)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Clean the extracted answer
            answer = matches[-1].strip()
            # Remove trailing punctuation but keep mathematical symbols
            answer = answer.rstrip('.,;:')
            return answer
    
    # Try to find any number at the end of the text
    number_pattern = r"(?:^|\s)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\d+/\d+)(?:\s*$|\s*[.,;]?\s*$)"
    matches = re.findall(number_pattern, text)
    if matches:
        return matches[-1].strip()
    
    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical expressions for comparison."""
    # Remove whitespace
    answer = answer.strip()
    answer = re.sub(r'\s+', '', answer)
    
    # Remove dollar signs
    answer = answer.replace('$', '')
    
    # Normalize LaTeX commands
    answer = answer.replace('\\left', '')
    answer = answer.replace('\\right', '')
    answer = answer.replace('\\Big', '')
    answer = answer.replace('\\big', '')
    answer = answer.replace('\\cdot', '*')
    answer = answer.replace('\\times', '*')
    answer = answer.replace('\\div', '/')
    
    # Handle fractions
    answer = normalize_fractions(answer)
    
    # Remove trailing punctuation
    answer = answer.rstrip('.,;:')
    
    return answer


def normalize_fractions(text: str) -> str:
    """Normalize fraction representations."""
    # Convert \frac{a}{b} to a/b for simple cases
    frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    
    def frac_replacer(match):
        num, den = match.groups()
        # For simple numeric fractions, compute the value
        try:
            num_val = float(eval(num))
            den_val = float(eval(den))
            if den_val != 0:
                result = num_val / den_val
                # Return as integer if it's a whole number
                if result == int(result):
                    return str(int(result))
                return str(result)
        except:
            pass
        return f"({num})/({den})"
    
    text = re.sub(frac_pattern, frac_replacer, text)
    
    # Also handle tfrac and dfrac
    text = text.replace('\\tfrac', '\\frac')
    text = text.replace('\\dfrac', '\\frac')
    
    return text


def is_equivalent(answer1: str, answer2: str) -> bool:
    """Check if two normalized answers are equivalent."""
    # Direct string comparison
    if answer1 == answer2:
        return True
    
    # Case-insensitive comparison for text answers
    if answer1.lower() == answer2.lower():
        return True
    
    # Check common mathematical equivalences
    equivalences = [
        ('infinity', '\\infty'),
        ('inf', '\\infty'),
        ('undefined', 'dne'),
        ('doesnotexist', 'dne'),
        ('none', 'dne'),
    ]
    
    a1_lower = answer1.lower()
    a2_lower = answer2.lower()
    
    for eq1, eq2 in equivalences:
        if (eq1 in a1_lower and eq2 in a2_lower) or (eq2 in a1_lower and eq1 in a2_lower):
            return True
    
    return False


def is_numerically_equivalent(answer1: str, answer2: str, tolerance: float = 1e-9) -> bool:
    """Check if two answers are numerically equivalent."""
    try:
        # Try to evaluate as numerical expressions
        val1 = evaluate_expression(answer1)
        val2 = evaluate_expression(answer2)
        
        if val1 is not None and val2 is not None:
            return abs(val1 - val2) < tolerance
    except:
        pass
    
    return False


def evaluate_expression(expr: str) -> Optional[float]:
    """Safely evaluate a mathematical expression."""
    try:
        # Remove common LaTeX commands that might remain
        expr = expr.replace('\\pi', '3.141592653589793')
        expr = expr.replace('\\e', '2.718281828459045')
        expr = expr.replace('^', '**')
        
        # Only allow safe operations
        allowed_names = {
            'abs': abs,
            'min': min,
            'max': max,
        }
        
        # Evaluate the expression safely
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return float(result)
    except:
        return None