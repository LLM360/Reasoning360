#!/usr/bin/env python3

import pandas as pd
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.nemotron_stem import compute_score, extract_solution


def test_extract_solution():
    """Test the extract_solution function with various response formats."""
    print("Testing extract_solution function...")
    
    test_cases = [
        ("The answer is A", "A"),
        ("Answer: B", "B"),
        ("\\boxed{C}", "C"),
        ("After careful analysis, the answer is D.", "D"),
        ("I think (C) is correct", "C"),
        ("No clear answer", None),
        ("The final answer is \\boxed{A}.", "A"),
        ("Answer: C", "C"),
    ]
    
    for response, expected in test_cases:
        result = extract_solution(response, method='strict')
        print(f"Input: '{response}' -> Expected: {expected}, Got: {result}")
        assert result == expected, f"Failed for '{response}': expected {expected}, got {result}"
    
    print("extract_solution tests passed!\n")


def test_compute_score():
    """Test the compute_score function."""
    print("Testing compute_score function...")
    
    # Test correct answer
    result = compute_score("Answer: A", "A")
    print(f"Correct answer test: {result}")
    assert result == {'score': 1.0, 'acc': 1.0}
    
    # Test incorrect answer
    result = compute_score("Answer: B", "A")
    print(f"Incorrect answer test: {result}")
    assert result == {'score': 0.0, 'acc': 0.0}
    
    # Test no extractable answer
    result = compute_score("I don't know", "A")
    print(f"No answer test: {result}")
    assert result == {'score': 0, 'acc': 0}
    
    print("compute_score tests passed!\n")


def test_default_compute_score():
    """Test the default_compute_score function with nemotron_stem data source."""
    print("Testing default_compute_score with nemotron_stem...")
    
    # Test with stem_nemotron data source
    result = default_compute_score("stem_nemotron", "Answer: C", "C")
    print(f"stem_nemotron correct: {result}")
    assert result == {'score': 1.0, 'acc': 1.0}
    
    result = default_compute_score("stem_nemotron", "Answer: A", "C")
    print(f"stem_nemotron incorrect: {result}")
    assert result == {'score': 0.0, 'acc': 0.0}
    
    # Test with nemotron_stem data source
    result = default_compute_score("nemotron_stem", "\\boxed{B}", "B")
    print(f"nemotron_stem correct: {result}")
    assert result == {'score': 1.0, 'acc': 1.0}
    
    print("default_compute_score tests passed!\n")


def test_real_data():
    """Test with real nemotron_stem data."""
    print("Testing with real nemotron_stem data...")
    
    try:
        # Load a sample of the test data
        df = pd.read_parquet('/mnt/sharefs/users/jianshu.she/nemotron_stem/test_data_final.parquet')
        sample = df.head(5)
        
        print(f"Testing with {len(sample)} samples from real data...")
        
        for idx, row in sample.iterrows():
            data_source = row['data_source']
            response = row['response']
            ground_truth = row['reward_model']['ground_truth']
            
            # Test with our implementation
            result = default_compute_score(data_source, response, ground_truth)
            print(f"Sample {idx}: response='{response}', ground_truth='{ground_truth}', score={result}")
            
        print("Real data test completed!\n")
        
    except Exception as e:
        print(f"Could not test with real data: {e}")


if __name__ == "__main__":
    test_extract_solution()
    test_compute_score()
    test_default_compute_score()
    test_real_data()
    print("All tests passed!")