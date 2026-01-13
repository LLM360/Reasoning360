#!/usr/bin/env python3
"""
Test script for DeepMath integration
"""

import sys
sys.path.append('/mnt/weka/home/jianshu.she/IFM/Reasoning360')

from verl.utils.reward_score import default_compute_score

def test_deepmath_scoring():
    """Test DeepMath scoring functionality"""
    
    print("Testing DeepMath scoring integration...")
    
    # Test cases
    test_cases = [
        {
            "solution": "\\boxed{0}",
            "ground_truth": "0",
            "expected": 1.0,
            "description": "Exact match with boxed answer"
        },
        {
            "solution": "The limit evaluates to 0",
            "ground_truth": "0",
            "expected": 1.0,
            "description": "Text extraction"
        },
        {
            "solution": "\\boxed{\\frac{2}{3}}",
            "ground_truth": "2/3",
            "expected": 1.0,
            "description": "Fraction equivalence"
        },
        {
            "solution": "\\boxed{42}",
            "ground_truth": "24",
            "expected": 0.0,
            "description": "Wrong answer"
        },
        {
            "solution": "The answer is \\infty",
            "ground_truth": "infinity",
            "expected": 1.0,
            "description": "Infinity equivalence"
        }
    ]
    
    print("\nRunning test cases:")
    print("=" * 60)
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        try:
            # Test with different data source identifiers
            for data_source in ["deepmath", "DeepMath", "zwhe99/DeepMath-103K"]:
                score = default_compute_score(
                    data_source=data_source,
                    solution_str=test["solution"],
                    ground_truth=test["ground_truth"]
                )
                
                passed = abs(score - test["expected"]) < 0.001
                
                if not passed:
                    print(f"❌ Test {i} FAILED ({data_source}): {test['description']}")
                    print(f"   Expected: {test['expected']}, Got: {score}")
                    all_passed = False
                    break
            else:
                print(f"✅ Test {i} PASSED: {test['description']}")
                
        except Exception as e:
            print(f"❌ Test {i} ERROR: {test['description']}")
            print(f"   Error: {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = test_deepmath_scoring()
    sys.exit(0 if success else 1)