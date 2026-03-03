#!/usr/bin/env python3
"""
Compute rewards for all parquet files in a directory.
Optionally compute response lengths using a specified tokenizer (default: DeepSeek-R1-0528).
"""

import argparse
import json
import os
import time
import logging

# Suppress pylatexenc warnings
logging.getLogger("pylatexenc").setLevel(logging.ERROR)

from datetime import datetime, timedelta
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from math import comb

from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import Dataset

from verl.utils.reward_score import default_compute_score

# Global tokenizer variable for response length computation
_tokenizer = None

def init_tokenizer(model_name: str = "deepseek-ai/DeepSeek-R1-0528"):
    """Initialize the tokenizer for response length computation."""
    global _tokenizer
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        print(f"Successfully initialized {model_name} tokenizer for length computation")
    except Exception as e:
        print(f"Warning: Failed to initialize tokenizer '{model_name}': {e}")
        _tokenizer = None

def compute_response_length(response: str) -> int:
    """Compute the token length of a response using the initialized tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        return 0
    return len(response) // 4
    # return len(_tokenizer.encode(response))

def compute_length_stats(lengths: List[int]) -> Dict[str, float]:
    """Compute min, max, and average length statistics."""
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths)
    }

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric using the unbiased estimator.
    
    pass@k estimates the probability that at least one of k samples passes.
    Formula: pass@k = E[1 - C(n-c, k) / C(n, k)]
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider (k <= n)
    
    Returns:
        pass@k probability (0.0 to 1.0)
    """
    if n == 0 or k == 0:
        return 0.0
    if c == 0:
        return 0.0
    if k > n:
        k = n
    
    # pass@k = 1 - C(n-c, k) / C(n, k)
    # If n-c < k, then C(n-c, k) = 0, so pass@k = 1.0
    if n - c < k:
        return 1.0
    
    try:
        return 1.0 - (comb(n - c, k) / comb(n, k))
    except (ValueError, OverflowError):
        # Handle edge cases
        return 0.0

def compute_pass_at_k_for_samples(scores: List[float], threshold: float, k_values: List[int]) -> Dict[str, float]:
    """
    Compute pass@k for multiple k values given a list of scores.
    
    Args:
        scores: List of scores for responses
        threshold: Threshold for considering a response as "passed"
        k_values: List of k values to compute pass@k for
    
    Returns:
        Dictionary mapping "pass@k" to computed values
    """
    n = len(scores)
    c = sum(1 for s in scores if s >= threshold)
    
    result = {}
    for k in k_values:
        if k <= n:
            result[f"pass@{k}"] = compute_pass_at_k(n, c, k)
    
    return result

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def sanitize_for_parquet(obj):
    """Recursively sanitize an object to make it parquet-compatible."""
    if obj is None:
        return None
    elif isinstance(obj, bool):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return int(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: sanitize_for_parquet(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_parquet(item) for item in obj]
    elif isinstance(obj, (str, int, float)):
        return obj
    else:
        return str(obj)

def compute_single_reward(arg_tuple):
    """Compute reward for one (response, ground-truth) pair."""
    gid, response, data_source, ground_truth, extra_info, resp_idx, compute_length, raw_resp = arg_tuple
    # print(f"Computing reward for {gid} with data_source {data_source}")
    # print(f"Response: {response}")
    # print(f"Ground truth: {ground_truth}")
    
    try:
        result = default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        
        if isinstance(result, dict):
            detailed = result
        else:
            detailed = {"score": float(result)}
            
        # Add response length if requested
        if compute_length:
            detailed["response_length"] = compute_response_length(raw_resp)
            
    except ValueError as e:
        if "embedded null byte" in str(e):
            detailed = {"score": 0.0, "error": "null_byte_in_response"}
            if compute_length:
                detailed["response_length"] = 0
            print(f"Warning: Null byte detected in response {gid}, marking as failed")
        else:
            # Re-raise other ValueError types
            raise
    except Exception as e:
        # Handle any other unexpected errors gracefully
        detailed = {"score": 0.0, "error": f"execution_error: {str(e)[:100]}"}
        if compute_length:
            detailed["response_length"] = 0
        print(f"Warning: Error computing reward for response {gid}: {str(e)[:100]}")
    
    detailed = sanitize_for_parquet(detailed)
    return gid, detailed, resp_idx

def read_parquet_file(file_path: str) -> pd.DataFrame:
    """Read a parquet file using HuggingFace datasets."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame()
    
    try:
        dataset = Dataset.from_parquet(file_path)
    except:
        print(f"Error reading parquet file: {file_path}")
        print("Trying to read parquet file with polars")
        import polars as pl
        dataset = pl.read_parquet(file_path)
    return dataset.to_pandas()

def write_parquet_file(file_path: str, data: pd.DataFrame):
    """Write DataFrame to a parquet file using HuggingFace datasets."""
    dataset = Dataset.from_pandas(data)
    dataset.to_parquet(file_path)

def process_parquet_file(file_path: str, output_path: str, args, reward_pool: Optional[Pool]) -> Tuple[int, float]:
    """Process a single parquet file and compute rewards."""
    print(f"Processing: {os.path.basename(file_path)}", flush=True)
    
    df = read_parquet_file(file_path)
    if df.empty:
        print(f"No valid data found in {file_path}", flush=True)
        return 0, 0.0
    
    start_time = time.time()
    tasks, lookup, gid = [], {}, 0
    
    # Track skipped items
    skipped = {"existing_scores": 0, "no_responses": 0, "no_ground_truth": 0}
    processed_rows = 0
    
    for i in range(len(df)):
        # Skip if already has scores and not recalculating
        current_scores = df.iloc[i].get("scores")
        if not args.recalculate_rewards and current_scores is not None:
            if isinstance(current_scores, (list, tuple)) and len(current_scores) > 0:
                skipped["existing_scores"] += 1
                continue
            elif isinstance(current_scores, np.ndarray) and current_scores.size > 0:
                skipped["existing_scores"] += 1
                continue
        
        responses = df.iloc[i][args.response_column_name]
        reward_model_data = df.iloc[i]["reward_model"]
        
        # Validate responses
        if responses is None or (isinstance(responses, float) and pd.isna(responses)):
            skipped["no_responses"] += 1
            continue
        
        if isinstance(responses, np.ndarray):
            responses = responses.tolist()
        
        if not isinstance(responses, list):
            skipped["no_responses"] += 1
            continue
        
        # Extract ground truth
        if not isinstance(reward_model_data, dict):
            skipped["no_ground_truth"] += 1
            continue
        
        if "logic__synlogic" in file_path:
            ground_truth = "XYZ"
        else:
            ground_truth = reward_model_data.get("ground_truth", "")
        
        # Handle different ground_truth formats
        if ground_truth is None:
            skipped["no_ground_truth"] += 1
            continue
        elif isinstance(ground_truth, str) and not ground_truth:
            skipped["no_ground_truth"] += 1
            continue
        elif isinstance(ground_truth, list) or (isinstance(ground_truth, np.ndarray) and ground_truth.ndim > 0):
            if len(ground_truth) == 0:
                skipped["no_ground_truth"] += 1
                continue
            # Convert ndarray to list while keeping original shape
            if isinstance(ground_truth, np.ndarray):
                ground_truth = ground_truth.tolist()
        elif isinstance(ground_truth, np.ndarray) and ground_truth.ndim == 0:
            # Handle numpy scalars (0-dimensional arrays)
            ground_truth = ground_truth.item()
        # For other types (int, float, etc.), use as-is
        
        # Get data source and extra info
        data_source = df.iloc[i].get("data_source", df.iloc[i].get("source", "unknown"))
        extra_info = df.iloc[i].get("extra_info", {})
        
        processed_rows += 1
        
        # Create tasks for each response
        for resp_idx, raw_resp in enumerate(responses):
            if raw_resp is None:
                continue
            
            # Extract response, removing thinking tags
            stripped = raw_resp.split("</think>", 1)[1] if "</think>" in str(raw_resp) else str(raw_resp)
            tasks.append((gid, stripped, data_source, ground_truth, extra_info, resp_idx, args.compute_response_length, raw_resp))
            lookup[gid] = i
            gid += 1
    
    if not tasks:
        print(f"No tasks to process in {os.path.basename(file_path)}")
        return 0, 0.0
    
    print(f"Computing rewards for {len(tasks)} responses across {processed_rows} items...", flush=True)
    
    # Initialize results
    detailed_by_sample: Dict[int, List[Optional[Dict]]] = {}
    for i in range(len(df)):
        responses = df.iloc[i][args.response_column_name]
        if isinstance(responses, (list, np.ndarray)):
            detailed_by_sample[i] = [None] * len(responses)
    
    # Process tasks
    if reward_pool:
        results = reward_pool.map(compute_single_reward, tasks)
    else:
        results = [compute_single_reward(task) for task in tqdm(tasks, desc="Computing rewards")]
    
    # Collect results
    for gidx, detailed, resp_idx in results:
        row_idx = lookup[gidx]
        if row_idx in detailed_by_sample:
            detailed_by_sample[row_idx][resp_idx] = detailed
    
    # Update DataFrame
    detailed_scores_list = [None] * len(df)
    scores_list = [None] * len(df)
    pass_rate_list = [None] * len(df)
    pass_at_k_list = [None] * len(df)
    response_lengths_list = [None] * len(df) if args.compute_response_length else None
    
    total_responses = 0
    total_passed = 0
    question_pass_rates = []
    
    # Length tracking
    all_response_lengths = []
    passed_response_lengths = []
    
    # Determine k values for pass@k based on available responses
    valid_response_counts = [len(df.iloc[i].get(args.response_column_name, [])) 
                            for i in range(len(df)) 
                            if isinstance(df.iloc[i].get(args.response_column_name), (list, np.ndarray))]
    max_responses = max(valid_response_counts) if valid_response_counts else 0
    k_values = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= max_responses]
    
    for row_idx, detailed_list in detailed_by_sample.items():
        # Fill missing results
        for i, d in enumerate(detailed_list):
            if d is None:
                missing_result = {"score": 0.0, "error": "missing"}
                if args.compute_response_length:
                    missing_result["response_length"] = 0
                detailed_list[i] = missing_result
        
        scores = [d["score"] for d in detailed_list]
        pass_cnt = sum(s >= args.correct_reward_threshold for s in scores)
        question_pass_rate = pass_cnt / len(scores) if len(scores) > 0 else 0.0
        
        # Compute pass@k for this sample
        pass_at_k_values = compute_pass_at_k_for_samples(scores, args.correct_reward_threshold, k_values)
        
        detailed_scores_list[row_idx] = detailed_list
        scores_list[row_idx] = scores
        pass_rate_list[row_idx] = question_pass_rate
        pass_at_k_list[row_idx] = pass_at_k_values
        
        # Extract response lengths if computed
        if args.compute_response_length:
            response_lengths = [d.get("response_length", 0) for d in detailed_list]
            response_lengths_list[row_idx] = response_lengths
            
            # Collect length statistics
            all_response_lengths.extend(response_lengths)
            for i, length in enumerate(response_lengths):
                if scores[i] >= args.correct_reward_threshold:
                    passed_response_lengths.append(length)
        
        question_pass_rates.append(question_pass_rate)
        total_passed += pass_cnt
        total_responses += len(scores)
    
    # Add results to DataFrame
    df["detailed_scores"] = detailed_scores_list
    df["scores"] = scores_list
    df["pass_rate"] = pass_rate_list
    df["pass_at_k"] = pass_at_k_list
    
    # Add response lengths if computed
    if args.compute_response_length:
        df["response_lengths"] = response_lengths_list
        
        # Compute and add length statistics
        all_length_stats = compute_length_stats(all_response_lengths)
        passed_length_stats = compute_length_stats(passed_response_lengths)
        
        length_stats = {
            "all_min": all_length_stats["min"],
            "all_max": all_length_stats["max"],
            "all_avg": all_length_stats["avg"],
            "passed_min": passed_length_stats["min"],
            "passed_max": passed_length_stats["max"],
            "passed_avg": passed_length_stats["avg"]
        }
        df["length_statistics"] = [length_stats for _ in range(len(df))]
    
    # Add model pass rate
    model_pass_rate = sum(question_pass_rates) / len(question_pass_rates) if len(question_pass_rates) > 0 else 0.0
    df["model_pass_rate"] = [{args.model_name: model_pass_rate} for _ in range(len(df))]
    
    # Compute aggregate pass@k statistics across all samples
    aggregate_pass_at_k = {f"pass@{k}": 0.0 for k in k_values}
    valid_samples = [val for val in pass_at_k_list if val is not None]
    if valid_samples:
        for k in k_values:
            key = f"pass@{k}"
            values = [sample.get(key, 0.0) for sample in valid_samples if key in sample]
            if values:
                aggregate_pass_at_k[key] = sum(values) / len(values)
    df["model_pass_at_k"] = [{args.model_name: aggregate_pass_at_k} for _ in range(len(df))]
    
    # Save results
    write_parquet_file(output_path, df)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"Results: {len(df)} items, {processed_rows} processed, {sum(skipped.values())} skipped")
    print(f"Pass rate: {model_pass_rate:.2%} ({total_passed}/{total_responses})")
    
    # Print pass@k statistics
    if aggregate_pass_at_k:
        # Sort by k value (extract integer from "pass@k")
        sorted_items = sorted(aggregate_pass_at_k.items(), key=lambda x: int(x[0].split('@')[1]))
        pass_at_k_str = ", ".join([f"{k}: {v:.2%}" for k, v in sorted_items])
        print(f"Pass@k: {pass_at_k_str}")
    
    pass_rate_output_path = "/".join(output_path.split("/")[:-1]) + "/pass_rate.jsonl"
    dataset_name = args.input_path.split("/")[-1]
    with open(pass_rate_output_path, "a") as f:
        f.write(json.dumps({
            "model_name": args.model_name,
            "dataset_name": dataset_name,
            "pass_rate": model_pass_rate,
            "total_responses": total_responses,
            "total_passed": total_passed,
            "processed_rows": processed_rows,
            "pass_at_k": aggregate_pass_at_k,
        }) + "\n")


    # Print response length statistics if computed
    if args.compute_response_length and all_response_lengths:
        all_stats = compute_length_stats(all_response_lengths)
        passed_stats = compute_length_stats(passed_response_lengths) if passed_response_lengths else {"min": 0, "max": 0, "avg": 0}
        print(f"Length stats - All: {all_stats['min']}/{all_stats['max']}/{all_stats['avg']:.1f}, Passed: {passed_stats['min']}/{passed_stats['max']}/{passed_stats['avg']:.1f} tokens")
    
    print(f"Time: {format_time(elapsed)}")
    
    return total_responses, elapsed

def main():
    parser = argparse.ArgumentParser(description="Compute rewards for parquet files")
    
    parser.add_argument("input_path", help="Directory or parquet file path")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--model_name", default="r1-0528", help="Model name")
    parser.add_argument("--output_suffix", default="", help="Output filename suffix")
    parser.add_argument("--pattern", default="*.parquet", help="File pattern for directories")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursive search")
    
    parser.add_argument("--reward_workers", type=int, default=64, help="Worker processes")
    parser.add_argument("--correct_reward_threshold", type=float, default=1.0, help="Pass threshold")
    parser.add_argument("--recalculate_rewards", action="store_true", help="Recompute existing rewards")
    parser.add_argument("--maxtasks_per_child", type=int, default=50, help="Tasks per worker")
    
    parser.add_argument("--compute_response_length", action="store_true", help="Compute response length using specified tokenizer")
    parser.add_argument("--tokenizer_model", default="deepseek-ai/DeepSeek-R1-0528", help="Tokenizer model name for response length computation (default: deepseek-ai/DeepSeek-R1-0528)")
    parser.add_argument("--response_column_name", default="r1_0528_responses", help="Column name for responses in the parquet file (default: r1_0528_responses)")
    parser.add_argument("--debug", action="store_true", help="Process first file only")
    
    args = parser.parse_args()
    
    # Initialize tokenizer if response length computation is requested
    if args.compute_response_length:
        init_tokenizer(args.tokenizer_model)
    
    # Validate input
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path does not exist: {args.input_path}")
        return
    
    is_single_file = input_path.is_file()
    
    # Setup output
    if args.output_dir:
        output_base = Path(args.output_dir)
        if is_single_file:
            stem = input_path.stem
            suffix = input_path.suffix
            output_filename = f"{stem}_{args.output_suffix}{suffix}" if args.output_suffix else f"{stem}_scored{suffix}"
            output_path = output_base / output_filename
        else:
            output_path = output_base
    else:
        if is_single_file:
            stem = input_path.stem
            suffix = input_path.suffix
            output_filename = f"{stem}_{args.output_suffix}{suffix}" if args.output_suffix else f"{stem}_scored{suffix}"
            output_path = input_path.parent / output_filename
        else:
            output_path = input_path.parent / f"{input_path.name}_scored"
    
    # Find files
    if is_single_file:
        parquet_files = [input_path]
    else:
        if args.recursive:
            parquet_files = list(input_path.rglob(args.pattern))
        else:
            parquet_files = list(input_path.glob(args.pattern))
        
        if not parquet_files:
            print(f"No files matching '{args.pattern}' found")
            return
        
        parquet_files.sort()
        
        if args.debug:
            parquet_files = parquet_files[:1]
    
    # Setup workers
    workers = min(args.reward_workers, max(1, cpu_count() - 1)) if args.reward_workers > 1 else 1
    reward_pool = Pool(processes=workers, maxtasksperchild=args.maxtasks_per_child) if workers > 1 else None
    
    print(f"Processing {len(parquet_files)} files with {workers} workers")
    print(f"Input: {args.input_path}")
    print(f"Output: {output_path}")
    
    # Create output directory
    if is_single_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process files
    total_processed = 0
    total_elapsed = 0.0
    successful_files = 0
    
    for file_path in parquet_files:
        if is_single_file:
            output_file_path = output_path
        else:
            relative_path = file_path.relative_to(input_path)
            if args.output_suffix:
                stem = relative_path.stem
                suffix = relative_path.suffix
                output_filename = f"{stem}_{args.output_suffix}{suffix}"
                output_file_path = output_path / relative_path.parent / output_filename
            else:
                output_file_path = output_path / relative_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if output file already exists when processing a folder
            if output_file_path.exists():
                print(f"Skipping {os.path.basename(file_path)} - output file already exists: {output_file_path}")
                continue
        
        processed, elapsed = process_parquet_file(str(file_path), str(output_file_path), args, reward_pool)
        total_processed += processed
        total_elapsed += elapsed
        successful_files += 1
            
    # Cleanup
    if reward_pool:
        reward_pool.close()
        reward_pool.join()
    
    # Summary
    print(f"\nComplete: {successful_files}/{len(parquet_files)} files")
    print(f"Total responses: {total_processed}")
    print(f"Total time: {format_time(total_elapsed)}")
    if total_processed > 0:
        print(f"Time per response: {total_elapsed/total_processed:.3f}s")

if __name__ == "__main__":
    main() 